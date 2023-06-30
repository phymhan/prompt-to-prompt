# Example of running (Prox-)MasaCtrl

if __name__ == "__main__":
    from prox_masactrl import main

    ########## MasaCtrl ##########
    main(
        out_dir="./outputs/masactrl_real/",
        source_image_path="./images/two_parrots.jpeg",
        source_prompt="Two parrots sitting on a stick with two steel bowls, city street background",
        target_prompt="Two kissing parrots sitting on a stick with two steel bowls, city street background",
        npi=False,
        npi_interp=0,
    )

    ########## ProxMasaCtrl ##########
    main(
        out_dir="./outputs/masactrl_real/",
        source_image_path="./images/two_parrots.jpeg",
        source_prompt="Two parrots sitting on a stick with two steel bowls, city street background",
        target_prompt="Two kissing parrots sitting on a stick with two steel bowls, city street background",
        inject_uncond="joint",
        npi=True,
        npi_interp=1,
    )  # OR
    main(
        out_dir="./outputs/masactrl_real/",
        source_image_path="./images/two_parrots.jpeg",
        source_prompt="Two parrots sitting on a stick with two steel bowls, city street background",
        target_prompt="Two kissing parrots sitting on a stick with two steel bowls, city street background",
        npi=True,
        npi_interp=1,
        prox='l0',
        quantile=0.6,
    )

    ########## MasaCtrl ##########
    main(
        out_dir="./outputs/masactrl_real/",
        source_image_path="./images/cake2.png",
        source_prompt="a round cake",
        target_prompt="a square cake",
        npi=False,
        npi_interp=0,
    )

    ########## ProxMasaCtrl ##########
    main(
        out_dir="./outputs/masactrl_real/",
        source_image_path="./images/cake2.png",
        source_prompt="a round cake",
        target_prompt="a square cake",
        npi=True,
        npi_interp=1,
        prox='l0',
        quantile=0.6,
    )
