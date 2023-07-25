# Description: Inversion of DDPM, modified from LEDITS
import torch
from torch import autocast, inference_mode
from inversion_utils import load_512, inversion_forward_process_elite, inversion_reverse_process_elite
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageFile
import torchvision.transforms as T
import os
import fire
from pipeline_elite import EliteGlobalPipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True


def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(
        0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs.to(torch.float32) / 2 + 0.5).clamp(0., 1.)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def add_margin(pil_img,
               top=0,
               right=0,
               bottom=0,
               left=0,
               color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def image_grid(imgs,
               rows=1,
               cols=None,
               size=None,
               titles=None,
               text_pos=(0, 0)):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if not size is None:
        imgs = [img.resize((size, size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows * cols

    top = 20
    w, h = imgs[0].size
    delta = 0
    if len(imgs) > 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    if not titles is None:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            size=20,
            encoding="unic")
        h = top + h
    grid = Image.new('RGB', size=(cols * w, rows * h + delta))
    for i, img in enumerate(imgs):

        if not titles is None:
            img = add_margin(img, top=top, bottom=0, left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i], (0, 0, 0), font=font)
        if not delta == 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


def invert(sd_pipe,
        x0: torch.FloatTensor,
        prompt_src: str = "",
        num_inference_steps=100,
        cfg_scale_src=3.5,
        eta=1,
        correlated_noise=False,
        ref_image=None,):

    #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion

    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps

    sd_pipe.scheduler.set_timesteps(num_inference_steps)

    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    # find Zs and wts - forward process
    wt, zs, wts = inversion_forward_process_elite(
        sd_pipe,
        w0,
        etas=eta,
        prompt=prompt_src,
        cfg_scale=cfg_scale_src,
        prog_bar=True,
        num_inference_steps=num_inference_steps,
        correlated_noise=correlated_noise,
        ref_image=ref_image,)
    return zs, wts


def sample(sd_pipe, zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta=1,
           prompt_null="", prox=None, quantile=0.7, ref_image=None,):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process_elite(
        sd_pipe,
        xT=wts[skip],
        etas=eta,
        prompts=[prompt_tar],
        prompts_null=[prompt_null],
        cfg_scales=[cfg_scale_tar],
        prog_bar=True,
        zs=zs[skip:],
        prox=prox,
        quantile=quantile,
        ref_image=ref_image,)

    # vae decode image
    with autocast("cuda"), inference_mode():
        x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]
    img = image_grid(x0_dec)
    return img


def main(
    sd_model_id="CompVis/stable-diffusion-v1-4",
    output_dir="./outputs/ddpm_inv_elite",
    input_image="./images/cat_mirror.png",
    ref_image="./images_elite/1.jpg",
    source_prompt="a cat sitting next to a mirror",
    target_prompt="a silver cat sculpture sitting next to a mirror",
    num_diffusion_steps=50,
    source_guidance_scale=1,
    skip_steps=0.36,
    target_guidance_scale=7.5,
    ddim_eta=1,
    correlated_noise=False,
    prox="none",
    quantile=0.75,
    npi=False,
    save_original=True,
    seed=42,
):

    if source_prompt != "":
        PREFIX = f"eta-{ddim_eta}_src-{source_guidance_scale}_tar-{target_guidance_scale}_skip-{skip_steps}"
    else:
        PREFIX = f"eta-{ddim_eta}_null-{source_guidance_scale}_tar-{target_guidance_scale}_skip-{skip_steps}"
    if prox == 'none' or prox == 'None':
        prox = None
    else:
        PREFIX += f"_{prox}-{quantile}"
    if npi:
        PREFIX += "_npi"
    skip_steps = int(skip_steps * num_diffusion_steps)
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))

    if isinstance(ref_image, str):
        ref_image = [ref_image]

    # load pipelines
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd_pipe = EliteGlobalPipeline.from_pretrained(
        sd_model_id, mapper_model_path='./checkpoints/global_mapper.pt').to(device)

    # uncomment for reproducabilty
    torch.manual_seed(seed)

    # Invert with ddpm
    x0 = load_512(input_image, device=device)
    # noise maps and latents
    zs, wts = invert(
        sd_pipe=sd_pipe,
        x0=x0,
        prompt_src=source_prompt,
        num_inference_steps=num_diffusion_steps,
        cfg_scale_src=source_guidance_scale,
        eta=ddim_eta,
        correlated_noise=correlated_noise,
        ref_image=ref_image,)
    ddpm_out_img = sample(
        sd_pipe,
        zs,
        wts,
        prompt_tar=target_prompt,
        skip=skip_steps,
        cfg_scale_tar=target_guidance_scale,
        eta=ddim_eta,
        prox=prox,
        quantile=quantile,
        prompt_null=source_prompt if npi else "",
        ref_image=ref_image,)

    # Show results
    orig_img_pt = load_512(input_image)
    orig_img = tensor_to_pil(orig_img_pt)[0]
    if save_original:
        output_image = image_grid([orig_img, ddpm_out_img], rows=1, cols=2)
    else:
        output_image = ddpm_out_img
    output_image.save(f"{output_dir}/{sample_count}_{PREFIX}_elite.png")


if __name__ == "__main__":
    fire.Fire(main)
