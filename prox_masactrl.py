import os
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.io import read_image
from diffusers import DDIMScheduler
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl, MutualSelfAttentionControlMaskAuto
import fire
from ptp_utils import slerp_tensor


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = T.Resize(512)(image)
    image = T.CenterCrop(512)(image)
    image = image.to(device)
    return image


def main(
    start_noise_interp: float = 0.0,
    model_path = "CompVis/stable-diffusion-v1-4",
    out_dir: str = "./outputs/masactrl_real/",
    source_image_path: str = "./images/statue-flower.png",
    source_prompt = "photo of a statue",
    target_prompt = "photo of a statue, side view",
    scale: float = 7.5,
    inv_scale: float = 1,
    query_intermediate: bool = False,
    masa_step: int = 4,
    masa_layer: int = 10,
    inject_uncond: str = "src",
    inject_cond: str = "src",
    prox_step: int = 0,
    prox: str = None,
    quantile: float = 0.7,
    npi: bool = False,
    npi_interp: float = 0,
    npi_step: int = 0,
    num_inference_steps: int = 50,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.5}).to(device)
    source_image = load_image(source_image_path, device)
    os.makedirs(out_dir, exist_ok=True)
    sample_count = len(os.listdir(out_dir))
    prompts = [source_prompt, target_prompt]

    # initialize the noise map
    # invert the source image
    start_code, latents_list = model.invert(source_image,
                                            source_prompt,
                                            guidance_scale=inv_scale,
                                            num_inference_steps=num_inference_steps,
                                            return_intermediates=True)
    if start_noise_interp > 0:
        random_code = model.prepare_latents(
            start_code.shape[0],
            start_code.shape[1], 
            512, 512, 
            dtype=start_code.dtype, 
            device=start_code.device, 
            generator=torch.Generator("cuda").manual_seed(42))
        start_code = torch.cat([
            start_code,
            slerp_tensor(start_noise_interp, start_code, random_code)
        ], dim=0)
    else:
        start_code = start_code.expand(len(prompts), -1, -1, -1)

    if prox == "none" or prox == "None":
        prox = None

    # hijack the attention module
    editor = MutualSelfAttentionControl(masa_step, masa_layer, inject_uncond=inject_uncond, inject_cond=inject_cond)
    # editor = MutualSelfAttentionControlMaskAuto(masa_step, masa_layer, ref_token_idx=1, cur_token_idx=2)  # NOTE: replace the token idx with the corresponding index in the prompt if needed
    regiter_attention_editor_diffusers(model, editor)

    if not query_intermediate:
        # inference the synthesized image
        image_masactrl = model(prompts,
                            latents=start_code,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=[1, scale],
                            neg_prompt=source_prompt if npi else None,
                            prox=prox,
                            prox_step=prox_step,
                            quantile=quantile,
                            npi_interp=npi_interp,
                            npi_step=npi_step,
        )
    else:
        # Note: querying the inversion intermediate features latents_list
        # may obtain better reconstruction and editing results
        image_masactrl = model(prompts,
                            latents=start_code,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=[1, scale],
                            neg_prompt=source_prompt if npi else None,
                            prox=prox,
                            prox_step=prox_step,
                            quantile=quantile,
                            npi_interp=npi_interp,
                            npi_step=npi_step,
                            ref_intermediate_latents=latents_list,
        )

    out_image = torch.cat([source_image * 0.5 + 0.5, image_masactrl], dim=0)
    npi = f"npi-{npi_step}-{npi_interp}" if npi else ""
    prox = f"prox-{prox_step}-{prox}-{quantile}" if prox else ""
    filename = f"{sample_count}_masa-step{masa_step}-layer{masa_layer}-u{inject_uncond}-c{inject_cond}_cfg-{scale}_{npi}_{prox}_noise-{start_noise_interp}.png"
    save_image(out_image, os.path.join(out_dir, filename))
    print("Syntheiszed images are saved in", os.path.join(out_dir, filename))
    print("Real image | Reconstructed image | Edited image")


if __name__ == "__main__":
    fire.Fire(main)
