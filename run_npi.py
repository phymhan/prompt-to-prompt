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


if __name__ == "__main__":
    from negative_prompt_inversion import main

    ########## NPI ##########
    main(
        image_path="./example_images/gnochi_mirror.jpeg",
        offsets=(0,0,200,0),
        prompt_src="a cat sitting next to a mirror",
        prompt_tar="a tiger sitting next to a mirror",
        output_dir="./outputs/cat_mirror",
        suffix="tiger",
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
        output_dir="./outputs/cat_mirror",
        suffix="tiger",
        guidance_scale=9,
        cross_replace_steps=0.7,
        self_replace_steps=0.6,
        blend_word=((('cat',), ("tiger",))),
        eq_params={"words": ("tiger",), "values": (2,)},
        is_replace_controller=True,
        proximal="l0",
        quantile=0.75,
        use_reconstruction_guidance=False,
    )

    ########## NPI ##########
    main(
        image_path="./images/cat_chair.png",
        prompt_src="A cat sitting on a wooden chair",
        prompt_tar="A dog sitting on a wooden chair",
        output_dir="./outputs/cat_chair",
        suffix="dog2",
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
        output_dir="./outputs/cat_chair",
        suffix="dog",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word="cat,dog",
        eq_params="dog,2",
        proximal="l0",
        quantile=0.75,
        use_reconstruction_guidance=True,
    )

    ########## ProxNPI ##########
    main(
        image_path="./images/coffee.jpeg",
        prompt_src="drawing of tulip on the coffee",
        prompt_tar="drawing of lion on the coffee",
        output_dir="./outputs/coffee",
        suffix="lion",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.5,
        blend_word="tulip,lion",
        eq_params="lion,2",
        proximal="l0",
        quantile=0.7,
    )
