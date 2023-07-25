import torch


if __name__ == "__main__":
    device = "cuda:0"
    from pipeline_elite import EliteGlobalPipeline
    from utils import load_images, image_grid
    use_fp16 = True
    bs = 4
    revision = "fp16" if use_fp16 else "fp32"
    pipe = EliteGlobalPipeline.from_pretrained(
        mapper_model_path='./checkpoints/global_mapper.pt',
        revision=revision,
    )
    pipe.to(device)
    ref_images = ["./images_elite/1.jpg"] * bs
    latents = torch.randn(
        (bs, 4, 64, 64), generator=torch.manual_seed(42),
    )
    syn_images = pipe(
        prompt=["a photo of a *"] * bs,
        placeholder_token='*',
        ref_image=ref_images,
        guidance_scale=5,
        eta=0,
        num_inference_steps=50,
        token_index="0",
        latents=latents,
    ).images
    syn_image = image_grid(syn_images, 1, bs)
    syn_image.save(f"test_elite_pipeline_{revision}.png")
