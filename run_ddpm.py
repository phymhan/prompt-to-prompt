# Example of running DDPM-Inversion and Proximal DDPM-Inversion

if __name__ == "__main__":
    from ddpm_inversion import main

    ########## DDPM-Inv ##########
    main(
        input_image="./images/cat_mirror.png",
        source_prompt="a cat sitting next to a mirror",
        target_prompt="a tiger sitting next to a mirror",
        source_guidance_scale=1,
        target_guidance_scale=7.5,
        skip_steps=0.36,
    )

    ########## Prox DDPM-Inv ##########
    main(
        input_image="./images/cat_mirror.png",
        source_prompt="a cat sitting next to a mirror",
        target_prompt="a tiger sitting next to a mirror",
        source_guidance_scale=1,
        target_guidance_scale=7.5,
        skip_steps=0.36,
        prox='l0',
        quantile=0.75,
    )

    # ---------- DDPM-Inv with ELITE ----------
    from ddpm_inv_elite import main
    """ The following are hardcoded in inversion_utils.encode_text_elite
        placeholder_token = "*"
        token_index = "0"
    """
    main(
        input_image="./images/cat1.png",
        ref_image="./images_elite/1.jpg",
        source_prompt="",
        target_prompt="a * sitting",
        source_guidance_scale=1,
        target_guidance_scale=7.5,
        skip_steps=0.36,
        prox='none',
    )

    main(
        input_image="./images/yellow_cat.jpeg",
        ref_image="./images_elite/20.jpg",
        source_prompt="",
        target_prompt="a * sitting on ground",
        source_guidance_scale=1,
        target_guidance_scale=7.5,
        skip_steps=0.36,
        prox='none',
    )
