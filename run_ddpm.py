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
