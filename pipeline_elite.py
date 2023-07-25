""" Modified from diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
"""
from typing import Callable, List, Optional, Union, Tuple, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask
import PIL
from PIL import Image
from utils import find_token_indices_batch
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            if idx is None:
                continue
            lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
            new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
            new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def inj_forward_crossattention(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
    if 'context' in kwargs:
        context = kwargs['context']  # NOTE: compatible with 0.10.0
    else:
        context = encoder_hidden_states
    if context is not None:
        context_tensor = context["CONTEXT_TENSOR"]
    else:
        context_tensor = hidden_states

    batch_size, sequence_length, _ = hidden_states.shape

    query = self.to_q(hidden_states)
    if context is not None:
        key = self.to_k_global(context_tensor)
        value = self.to_v_global(context_tensor)
    else:
        key = self.to_k(context_tensor)
        value = self.to_v(context_tensor)

    dim = query.shape[-1]

    query = self.reshape_heads_to_batch_dim(query)
    key = self.reshape_heads_to_batch_dim(key)
    value = self.reshape_heads_to_batch_dim(value)

    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores * self.scale

    attention_probs = attention_scores.softmax(dim=-1)

    hidden_states = torch.matmul(attention_probs, value)
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    return hidden_states


class EliteGlobalPipeline(StableDiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        image_encoder: CLIPVisionModel,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        mapper: Mapper,
        scheduler: Union[
            DDIMScheduler,
            LMSDiscreteScheduler,
        ],
        requires_safety_checker: bool = False,
    ):
        # super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            mapper=mapper,
            scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.image_transform_clip = self.get_image_transform_clip()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4', **kwargs):
        replace_ca_forward = not kwargs.get("no_replace_ca_forward", False)
        local_path_only = pretrained_model_name_or_path is not None
        dtype = torch.float16 if kwargs.get("revision", "fp32") == "fp16" else torch.float32
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_path_only,
        )

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)

        # Load models and create wrapper for stable diffusion
        for _module in text_encoder.modules():
            if _module.__class__.__name__ == "CLIPTextTransformer":
                _module.__class__.__call__ = inj_forward_text

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=dtype,
            local_files_only=local_path_only,
        )

        mapper = Mapper(input_dim=1024, output_dim=768)
        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == "CrossAttention":
                if 'attn1' in _name: continue
                if replace_ca_forward:
                    _module.__class__.__call__ = inj_forward_crossattention

                shape = _module.to_k.weight.shape
                to_k_global = nn.Linear(shape[1], shape[0], bias=False)
                mapper.add_module(f'{_name.replace(".", "_")}_to_k', to_k_global)

                shape = _module.to_v.weight.shape
                to_v_global = nn.Linear(shape[1], shape[0], bias=False)
                mapper.add_module(f'{_name.replace(".", "_")}_to_v', to_v_global)
        mapper.load_state_dict(torch.load(kwargs.get('mapper_model_path', './checkpoints/global_mapper.pt'), map_location='cpu'))
        if dtype == torch.float16:
            mapper.half()
        mapper.dtype = dtype

        for _name, _module in unet.named_modules():
            if 'attn1' in _name: continue
            if _module.__class__.__name__ == "CrossAttention":
                _module.add_module('to_k_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
                _module.add_module('to_v_global', mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
        )
        vae.eval()
        unet.eval()
        image_encoder.eval()
        text_encoder.eval()
        mapper.eval()

        init_kwargs = {
            "vae": vae,
            "unet": unet,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "mapper": mapper,
            "scheduler": scheduler,
            "requires_safety_checker": False,
        }
        model = cls(**init_kwargs)
        return model
    
    def to(self, device):
        super().to(device)
        self.mapper.device = device
        return self

    def decode_latents(self, latents, to_numpy=True):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        if to_numpy:
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def encode_images(self, images, generator=None):
        images = 2.0 * images - 1.0
        latents = self.vae.encode(images).latent_dist.sample(generator)
        latents = 0.18215 * latents
        return latents
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(dtype).to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def get_image_transform_clip(self, normalize=True, toTensor=True):
        transform_list = []
        if toTensor:
            transform_list += [torchvision.transforms.ToTensor()]
        if normalize:
            transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                (0.26862954, 0.26130258, 0.27577711))]
        return torchvision.transforms.Compose(transform_list)

    def process_images_clip(self, images_pil, size=224):
        # NOTE: assumes image is a list of PIL images
        image_tensor = []
        for image_pil in images_pil:
            if isinstance(image_pil, str):
                image_pil = PIL.Image.open(image_pil).convert("RGB")
            image_pil = image_pil.resize((size, size), resample=PIL.Image.BICUBIC)
            image_tensor.append(self.image_transform_clip(image_pil))
        return torch.stack(image_tensor)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        ref_image: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        placeholder_token: Optional[str] = "*",
        token_index: Optional[str] = "0",
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)
        if isinstance(prompt, str):
            prompt = [prompt]
        # if isinstance(negative_prompt, str):
        #     negative_prompt = [negative_prompt]
        if isinstance(ref_image, str):
            ref_image = [ref_image]

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        dtype = next(self.unet.parameters()).dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt = [p.format(placeholder_token) for p in prompt]
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        ref_image = self.process_images_clip(ref_image)
        ref_image = ref_image.to(dtype).to(device)
        image_features = self.image_encoder(ref_image, output_hidden_states=True)
        image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                            image_features[2][16]]
        image_embeddings = [emb.detach() for emb in image_embeddings]
        inj_embedding = self.mapper(image_embeddings)  # [batch_size, 5, 768]
        if token_index != 'full':  # NOTE: truncate inj_embedding
            if ':' in token_index:
                token_index = token_index.split(':')
                token_index = slice(int(token_index[0]), int(token_index[1]))
            else:
                token_index = slice(int(token_index), int(token_index) + 1)
            inj_embedding = inj_embedding[:, token_index, :]
        placeholder_idx = find_token_indices_batch(self.tokenizer, prompt, placeholder_token)
        encoder_hidden_states = self.text_encoder({
            "input_ids": input_ids,
            "inj_embedding": inj_embedding,
            "inj_index": placeholder_idx})[0]
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder({'input_ids': uncond_input.input_ids.to(device)})[0]

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states={
                        "CONTEXT_TENSOR": encoder_hidden_states,
                    }
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states={
                            "CONTEXT_TENSOR": uncond_embeddings,
                        }
                    ).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        has_nsfw_concept = [False] * len(image)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
