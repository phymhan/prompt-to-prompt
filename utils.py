from PIL import Image


def load_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    images_pil = []
    for image_path in image_paths:
        if isinstance(image_path, str):
            image_pil = Image.open(image_path).convert("RGB")
        else:
            assert isinstance(image_path, Image)
            image_pil = image_path
        images_pil.append(image_pil)
    return images_pil


def image_grid(imgs, rows, cols):
    # copied from https://huggingface.co/blog/stable_diffusion
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def find_token_indices(tokenizer, prompt, pattern, return_decoded=False):
    if pattern not in prompt:
        return [None]
    idx_in_str = prompt.index(pattern)
    tokens = tokenizer(prompt)['input_ids']
    pre_pattern_str = prompt[:idx_in_str]
    pre_pattern_tokens = tokenizer(pre_pattern_str)['input_ids']
    pattern_tokens = tokenizer(pattern)['input_ids']
    num_pattern_tokens = len(pattern_tokens) - 2
    token_indices = list(range(len(pre_pattern_tokens) - 2, len(pre_pattern_tokens) - 2 + num_pattern_tokens))
    token_indices = [i + 1 for i in token_indices]
    if return_decoded:
        token_idx_to_word = {idx: tokenizer.decode(t) for idx, t in enumerate(tokens) if 0 < idx < len(tokens) - 1}
        word = ' '.join([token_idx_to_word[i] for i in token_indices])
        return token_indices, word
    return token_indices


def find_token_indices_batch(tokenizer, prompts, patterns):
    # NOTE: hardcoded to return the first index for each entry in the batch
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(patterns, str):
        patterns = [patterns] * len(prompts)
    token_indices_batch = []
    for prompt, pattern in zip(prompts, patterns):
        token_indices = find_token_indices(tokenizer, prompt, pattern)
        token_indices_batch.append(token_indices[0])
    return token_indices_batch
