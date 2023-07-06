# Example of running NPI and ProxNPI

if __name__ == "__main__":
    from negative_prompt_inversion import main

    ########## NPI ##########
    main(
        image_path="./example_images/gnochi_mirror.jpeg",
        offsets=(0,0,200,0),
        prompt_src="a cat sitting next to a mirror",
        prompt_tar="a tiger sitting next to a mirror",
        output_dir="./outputs/npi_real/",
        suffix="cat2tiger",
        guidance_scale=9,
        cross_replace_steps=0.7,
        self_replace_steps=0.6,
        blend_word=((('cat',), ("tiger",))),
        eq_params={"words": ("tiger",), "values": (2,)},
        is_replace_controller=True,
        proximal=None,
    )

    ########## ProxNPI ##########
    main(
        image_path="./example_images/gnochi_mirror.jpeg",
        offsets=(0,0,200,0),
        prompt_src="a cat sitting next to a mirror",
        prompt_tar="a tiger sitting next to a mirror",
        output_dir="./outputs/npi_real/",
        suffix="cat2tiger",
        guidance_scale=9,
        cross_replace_steps=0.7,
        self_replace_steps=0.6,
        blend_word=((('cat',), ("tiger",))),
        eq_params={"words": ("tiger",), "values": (2,)},
        is_replace_controller=True,
        proximal="l0",
        quantile=0.75,
        use_inversion_guidance=True,
        recon_lr=1,
        recon_t=400,
        dilate_mask=2,
    )

    ########## NPI ##########
    main(
        image_path="./images/cat_chair.png",
        prompt_src="A cat sitting on a wooden chair",
        prompt_tar="A dog sitting on a wooden chair",
        output_dir="./outputs/npi_real/",
        suffix="cat2dog",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word="cat,dog",
        eq_params="dog,2",
        proximal=None,
    )

    ########## ProxNPI ##########
    main(
        image_path="./images/cat_chair.png",
        prompt_src="A cat sitting on a wooden chair",
        prompt_tar="A dog sitting on a wooden chair",
        output_dir="./outputs/npi_real/",
        suffix="cat2dog",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word="cat,dog",
        eq_params="dog,2",
        proximal="l0",
        quantile=0.75,
        use_inversion_guidance=True,
        recon_lr=1,
        recon_t=400,
    )

    ########## ProxNPI ##########
    main(
        image_path="./images/coffee.jpeg",
        prompt_src="drawing of tulip on the coffee",
        prompt_tar="drawing of lion on the coffee",
        output_dir="./outputs/npi_real/",
        suffix="tulip2lion",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.5,
        blend_word="tulip,lion",
        eq_params="lion,2",
        proximal="l0",
        quantile=0.7,
    )

    # More ProxNPI examples
    main(
        image_path="./images/tiger_white.png",
        prompt_src="white tiger on brown ground",
        prompt_tar="white cat on brown ground",
        output_dir="./outputs/npi_real/",
        suffix="tiger2cat",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word="tiger,cat",
        eq_params="cat,2",
        proximal="l0",
        quantile=0.7,
        use_inversion_guidance=True,
        recon_lr=0.5,
        recon_t=600,
    )

    main(
        image_path="./images/plate_fruit.jpeg",
        prompt_src="white plate with fruits on it",
        prompt_tar="white plate with pizza on it",
        output_dir="./outputs/npi_real/",
        suffix="fruit2pizza",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=((('fruits',), ("pizza",))),
        eq_params={"words": ("pizza",), "values": (2,)},
        proximal="l0",
        quantile=0.7,
        use_inversion_guidance=True,
        recon_lr=0.5,
        recon_t=600,
    )

    main(
        image_path="./images/woman_blue_hair.png",
        prompt_src="A woman with blue hair  in the forest",
        prompt_tar="A storm-trooper with blue hair  in the forest",
        output_dir="./outputs/npi_real/",
        suffix="woman2stormtrooper",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=((('woman',), ("storm-trooper",))),
        eq_params={"words": ("storm-trooper",), "values": (2,)},
        proximal="l0",
        quantile=0.75,
        use_inversion_guidance=True,
        recon_lr=0.5,
        recon_t=600,
    )

    main(
        image_path="./images/van.jpg",
        prompt_src="orange van with surfboards on top",
        prompt_tar="orange van with flowers on top",
        output_dir="./outputs/npi_real/",
        suffix="surfboards2flowers",
        guidance_scale=9,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=((('surfboards',), ("flowers",))),
        eq_params={"words": ("flowers", ), "values": (2,)},
        proximal='l0',
        quantile=0.9,
        use_inversion_guidance=True,
        recon_lr=0.1,
        recon_t=1000,
    )
