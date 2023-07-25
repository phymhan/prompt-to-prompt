# %% [markdown]
# ## Copyright 2022 Google LLC. Double-click for license information.

# %%
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Null-text inversion + Editing with Prompt-to-Prompt

# %%
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import os
from scheduler_dev import DDIMSchedulerDev
from utils import find_token_indices_batch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
MAX_NUM_WORDS = 77
LATENT_SIZE = (64, 64)


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,
                   inference_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=400,
                   inversion_guidance=False, x_stars=None, i=0, **kwargs):
    bs = latents.shape[0]
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(
        latents_input,
        t,
        encoder_hidden_states={"CONTEXT_TENSOR": context})["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    step_kwargs = {
        'ref_image': None,
        'recon_lr': 0,
        'recon_mask': None,
    }
    mask_edit = None
    if inference_stage and prox is not None:
        if prox == 'l1':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                if kwargs.get('dilate_mask', 0) > 0:
                    radius = int(kwargs.get('dilate_mask', 0))
                    mask_edit = ptp_utils.dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        elif prox == 'l0':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                if kwargs.get('dilate_mask', 0) > 0:
                    radius = int(kwargs.get('dilate_mask', 0))
                    mask_edit = ptp_utils.dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        else:
            raise NotImplementedError
        noise_pred = noise_pred_uncond + guidance_scale * score_delta
    else:
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **step_kwargs)["prev_sample"]
    if mask_edit is not None and inversion_guidance and (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
        recon_mask = 1 - mask_edit
        latents = latents - recon_lr * (latents - x_stars[len(x_stars)-i-2].expand_as(latents)) * recon_mask

    latents = controller.step_callback(latents)
    return latents


class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=LATENT_SIZE)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:

            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th = th
        
        
class EmptyControl:

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer=None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


def make_controller(pipeline, prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer=pipeline.tokenizer)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer=pipeline.tokenizer)
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NegativePromptInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(
            latents, 
            t, 
            encoder_hidden_states={"CONTEXT_TENSOR": context})["sample"]
        return noise_pred

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder({'input_ids': uncond_input.input_ids.to(self.model.device)})[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder({'input_ids': text_input.input_ids.to(self.model.device)})[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * NUM_DDIM_STEPS
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


# %% [markdown]
# ## Infernce Code

# %%
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    inference_stage=True,
    prox=None,
    quantile=0.7,
    image_enc=None,
    recon_lr=0.1,
    recon_t=400,
    inversion_guidance=False,
    x_stars=None,
    **kwargs,
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    if os.path.exists(kwargs.get('ref_image_path', "")):
        token_index = kwargs.get('token_index', '0')
        input_ids = model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(model.device)
        ref_image = kwargs['ref_image_path']
        ref_image = model.process_images_clip([ref_image])
        ref_image = ref_image.to(device)
        image_features = model.image_encoder(ref_image, output_hidden_states=True)
        image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                            image_features[2][16]]
        image_embeddings = [emb.detach() for emb in image_embeddings]
        inj_embedding = model.mapper(image_embeddings)  # [1, 5, 768]
        if token_index != 'full':  # NOTE: truncate inj_embedding
            if ':' in token_index:
                token_index = token_index.split(':')
                token_index = slice(int(token_index[0]), int(token_index[1]))
            else:
                token_index = slice(int(token_index), int(token_index) + 1)
            inj_embedding = inj_embedding[:, token_index, :]
        inj_embedding = torch.cat([inj_embedding, inj_embedding], dim=0)  # NOTE: a hack for ptp only
        placeholder_idx = find_token_indices_batch(model.tokenizer, prompt, "S")
        text_embeddings = model.text_encoder({
            "input_ids": input_ids,
            "inj_embedding": inj_embedding,
            "inj_index": placeholder_idx})[0]
    else:
        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder({'input_ids': text_input.input_ids.to(model.device)})[0]
    max_length = model.tokenizer.model_max_length
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder({'input_ids': uncond_input.input_ids.to(model.device)})[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource=False,
            inference_stage=inference_stage, prox=prox, quantile=quantile,
            image_enc=image_enc, recon_lr=recon_lr, recon_t=recon_t,
            inversion_guidance=inversion_guidance, x_stars=x_stars, i=i, **kwargs)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(pipeline, prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True,
                    inference_stage=True, prox=None, quantile=0.7, image_enc=None, recon_lr=0.1, recon_t=400, guidance_scale=7.5, 
                    inversion_guidance=False, x_stars=None, **kwargs):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(pipeline, prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(pipeline, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS,
                                        guidance_scale=guidance_scale, generator=generator, uncond_embeddings=uncond_embeddings,
                                        inference_stage=inference_stage, prox=prox, quantile=quantile,
                                        image_enc=image_enc, recon_lr=recon_lr, recon_t=recon_t,
                                        inversion_guidance=inversion_guidance, x_stars=x_stars, **kwargs)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t


# %%
def main(
        image_path,
        ref_image_path,
        prompt_src,
        prompt_tar,
        output_dir='output',
        suffix='edit1',
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        offsets=(0,0,0,0),
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
        token_index='0',
):
    GUIDANCE_SCALE = guidance_scale
    PROXIMAL = proximal
    QUANTILE = quantile
    RECON_T = recon_t
    RECON_LR = recon_lr

    if PROXIMAL == 'none' or PROXIMAL == 'None':
        PROXIMAL = None

    if PROXIMAL is None:
        PREFIX = f'npi-{GUIDANCE_SCALE}'
    else:
        PREFIX = f'prox-{PROXIMAL}-{QUANTILE}-{GUIDANCE_SCALE}'

    if (PROXIMAL is not None) and use_reconstruction_guidance:
        PREFIX = f'{PREFIX}-rec-{RECON_T}-{RECON_LR}'
    
    if (PROXIMAL is not None) and use_inversion_guidance:
        PREFIX = f'{PREFIX}-inv-{RECON_T}-{RECON_LR}'
    
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))
    PREFIX = f'{sample_count}_{PREFIX}'

    scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
    from pipeline_elite import EliteGlobalPipeline
    ldm_stable = EliteGlobalPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        mapper_model_path='./checkpoints/global_mapper.pt',
        no_replace_ca_forward=True).to(device)
    ldm_stable.scheduler = scheduler

    null_inversion = NegativePromptInversion(ldm_stable)
    (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = null_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)
    x_t = x_stars[-1]
    if npi_interp > 0:
        PREFIX = f'{PREFIX}-interp-{npi_interp}'
    if not os.path.exists(f"{output_dir}/image-gt.png"):
        Image.fromarray(image_gt).save(f"{output_dir}/image-gt.png")

    # %%
    prompts = [prompt_src]
    controller = AttentionStore()
    image_inv, x_t = run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False,
                                    inference_stage=False, guidance_scale=GUIDANCE_SCALE)
    print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")

    inference_kwargs = {
        'inference_stage': True,
        'prox': PROXIMAL,
        'quantile': QUANTILE,
        'image_enc': image_enc_latent if use_reconstruction_guidance else None,
        'recon_lr': RECON_LR if use_reconstruction_guidance or use_inversion_guidance else 0,
        'recon_t': RECON_T if use_reconstruction_guidance or use_inversion_guidance else 1000,
        'inversion_guidance': use_inversion_guidance,
        'x_stars': x_stars,
        'dilate_mask': dilate_mask,
        'ref_image_path': ref_image_path,
        'token_index': token_index,
    }

    ########## edit ##########
    prompts = [prompt_src, prompt_tar]
    cross_replace_steps = {'default_': cross_replace_steps,}
    if isinstance(blend_word, str):
        s1, s2 = blend_word.split(",")
        blend_word = (((s1,), (s2,))) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
    if isinstance(eq_params, str):
        s1, s2 = eq_params.split(",")
        eq_params = {"words": (s1,), "values": (float(s2),)} # amplify attention to the word "tiger" by *2 
    controller = make_controller(ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps, blend_word, eq_params)
    images, _ = run_and_display(ldm_stable, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,
                                guidance_scale=GUIDANCE_SCALE, **inference_kwargs)

    Image.fromarray(np.concatenate(images, axis=1)).save(f"{output_dir}/{PREFIX}_{suffix}.png")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
