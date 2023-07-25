import torch
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import torchvision.transforms as T
import os
import yaml
import numpy as np


def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16)

    return image


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) /
            (1 - alpha_bar)) * x0 + ((alpha_t**0.5 * (1 - alpha_prod_t_prev)) /
                                     (1 - alpha_bar)) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar)**0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (num_inference_steps, model.unet.in_channels,
                            model.unet.sample_size, model.unet.sample_size)
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device, dtype=torch.float16)
    for t in reversed(timesteps):  # 1...T
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t]**0.5) + torch.randn_like(
            x0, dtype=torch.float16) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0], dim=0)
    return xts


def sample_xts_from_x0_mc(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar)**0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (num_inference_steps, model.unet.in_channels,
                            model.unet.sample_size, model.unet.sample_size)
    iid_noises = torch.randn(len(model.scheduler.alphas), *variance_noise_shape[1:], dtype=torch.float16).to(x0.device)
    iid_noises = iid_noises * ((betas / alpha_bar)**0.5).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(x0.device)
    iid_noises_cumsum = iid_noises.cumsum(dim=0)
    noises = iid_noises_cumsum * ((alpha_bar / (1 - alpha_bar))**0.5).unsqueeze(1).unsqueeze(2).unsqueeze(3).to(x0.device)
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device, dtype=torch.float16)
    for t in reversed(timesteps):  # 1...T
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t]**0.5) + noises[t].unsqueeze(0) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0], dim=0)
    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(
            text_input.input_ids.to(model.device))[0]
    return text_encoding


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep + model.scheduler.config.num_train_timesteps //
        model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t**
                            (0.5) * model_output) / alpha_prod_t**(0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample, model_output,
                                            torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep):  #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev /
                beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


@torch.no_grad()
def inversion_forward_process(model,
                              x0,
                              etas=None,
                              prog_bar=False,
                              prompt="",
                              cfg_scale=3.5,
                              num_inference_steps=50,
                              eps=None,
                              correlated_noise=False,):

    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (num_inference_steps, model.unet.in_channels,
                            model.unet.sample_size, model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        if not correlated_noise:
            xts = sample_xts_from_x0(
                model,
                x0,
                num_inference_steps=num_inference_steps)
        else:
            xts = sample_xts_from_x0_mc(
                model,
                x0,
                num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape,
                         device=model.device,
                         dtype=torch.float16)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps),
              desc="Inverting...") if prog_bar else reversed(timesteps)

    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        out = model.unet.forward(xt,
                                 timestep=t,
                                 encoder_hidden_states=uncond_embedding)
        if not prompt == "":
            cond_out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=text_embeddings)

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample -
                                                   out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t])**0.5 * noise_pred) / alpha_bar[t]**0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev -
                                     etas[idx] * variance)**(0.5) * noise_pred

            mu_xt = alpha_prod_t_prev**(
                0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts


def reverse_step(model,
                 model_output,
                 timestep,
                 sample,
                 eta=0,
                 variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t**
                            (0.5) * model_output) / alpha_prod_t**(0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep)  #, prev_timestep)
    std_dev_t = eta * variance**(0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev -
                             eta * variance)**(0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev**(
        0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape,
                                         device=model.device,
                                         dtype=torch.float16)
        sigma_z = eta * variance**(0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


@torch.no_grad()
def inversion_reverse_process(model,
                              xT,
                              etas=0,
                              prompts="",
                              prompts_null="",
                              cfg_scales=None,
                              prog_bar=False,
                              zs=None,
                              controller=None,
                              asyrp=False,
                              prox=None,
                              quantile=0.7):

    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(
        model.device, dtype=torch.float16)

    text_embeddings = encode_text(model, prompts)
    # uncond_embedding = encode_text(model, [""] * batch_size)
    uncond_embedding = encode_text(model, prompts_null)

    if etas is None: etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(
        timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = t_to_idx[int(t)]
        ## Unconditional embedding
        uncond_out = model.unet.forward(xt,
                                        timestep=t,
                                        encoder_hidden_states=uncond_embedding)

        ## Conditional embedding
        if prompts:
            cond_out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=text_embeddings)

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            if prox == 'l0' or prox == 'l1':
                score_delta = cond_out.sample - uncond_out.sample
                threshold = score_delta.abs().quantile(quantile)
                score_delta -= score_delta.clamp(-threshold, threshold)
                if prox == 'l1':
                    score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                    score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                noise_pred = uncond_out.sample + cfg_scales_tensor * score_delta
            else:
                noise_pred = uncond_out.sample + cfg_scales_tensor * (
                    cond_out.sample - uncond_out.sample)
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model,
                          noise_pred,
                          t,
                          xt,
                          eta=etas[idx],
                          variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)
    return xt, zs


""" Modified for ELITE
"""
from utils import find_token_indices_batch

@torch.no_grad()
def encode_text_elite(model, prompts, ref_images=None, token_index='0'):
    if ref_images is not None:
        input_ids = model.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(model.device)
        ref_images = model.process_images_clip(ref_images)
        ref_images = ref_images.to(model.device)
        image_features = model.image_encoder(ref_images, output_hidden_states=True)
        image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                            image_features[2][16]]
        image_embeddings = [emb.detach() for emb in image_embeddings]
        inj_embedding = model.mapper(image_embeddings)  # [batch_size, 5, 768]
        if token_index != 'full':  # NOTE: truncate inj_embedding
            if ':' in token_index:
                token_index = token_index.split(':')
                token_index = slice(int(token_index[0]), int(token_index[1]))
            else:
                token_index = slice(int(token_index), int(token_index) + 1)
            inj_embedding = inj_embedding[:, token_index, :]
        placeholder_idx = find_token_indices_batch(model.tokenizer, prompts, "*")
        text_encoding = model.text_encoder({
            "input_ids": input_ids,
            "inj_embedding": inj_embedding,
            "inj_index": placeholder_idx})[0]
    else:
        uncond_input = model.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_encoding = model.text_encoder({'input_ids': uncond_input.input_ids.to(model.device)})[0]
    return text_encoding


@torch.no_grad()
def inversion_reverse_process_elite(
    model,
    xT,
    etas=0,
    prompts="",
    prompts_null="",
    ref_image=None,
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    asyrp=False,
    prox=None,
    quantile=0.7
):

    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(
        model.device, dtype=torch.float16)

    text_embeddings = encode_text_elite(model, prompts, ref_image)
    uncond_embedding = encode_text_elite(model, prompts_null, None)

    if etas is None: etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(
        timesteps[-zs.shape[0]:]) if prog_bar else timesteps[-zs.shape[0]:]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = t_to_idx[int(t)]
        ## Unconditional embedding
        uncond_out = model.unet.forward(
            xt,
            timestep=t,
            encoder_hidden_states={
                "CONTEXT_TENSOR": uncond_embedding,
            }
        )

        ## Conditional embedding
        if prompts:
            cond_out = model.unet.forward(
                xt,
                timestep=t,
                encoder_hidden_states={
                    "CONTEXT_TENSOR": text_embeddings,
                }
            )

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            if prox == 'l0' or prox == 'l1':
                score_delta = cond_out.sample - uncond_out.sample
                threshold = score_delta.abs().quantile(quantile)
                score_delta -= score_delta.clamp(-threshold, threshold)
                if prox == 'l1':
                    score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
                    score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
                noise_pred = uncond_out.sample + cfg_scales_tensor * score_delta
            else:
                noise_pred = uncond_out.sample + cfg_scales_tensor * (
                    cond_out.sample - uncond_out.sample)
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model,
                          noise_pred,
                          t,
                          xt,
                          eta=etas[idx],
                          variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)
    return xt, zs


@torch.no_grad()
def inversion_forward_process_elite(
    model,
    x0,
    etas=None,
    prog_bar=False,
    prompt="",
    ref_image=None,
    cfg_scale=3.5,
    num_inference_steps=50,
    eps=None,
    correlated_noise=False,
):
    if not prompt == "":
        text_embeddings = encode_text_elite(model, prompt, ref_image)
    uncond_embedding = encode_text_elite(model, "", None)
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (num_inference_steps, model.unet.in_channels,
                            model.unet.sample_size, model.unet.sample_size)
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        if not correlated_noise:
            xts = sample_xts_from_x0(
                model,
                x0,
                num_inference_steps=num_inference_steps)
        else:
            xts = sample_xts_from_x0_mc(
                model,
                x0,
                num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape,
                         device=model.device,
                         dtype=torch.float16)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps),
              desc="Inverting...") if prog_bar else reversed(timesteps)

    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        out = model.unet.forward(
            xt,
            timestep=t,
            encoder_hidden_states={"CONTEXT_TENSOR": uncond_embedding}
        )
        if not prompt == "":
            cond_out = model.unet.forward(
                xt,
                timestep=t,
                encoder_hidden_states={"CONTEXT_TENSOR": text_embeddings}
            )

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample -
                                                   out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t])**0.5 * noise_pred) / alpha_bar[t]**0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev -
                                     etas[idx] * variance)**(0.5) * noise_pred

            mu_xt = alpha_prod_t_prev**(
                0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts
