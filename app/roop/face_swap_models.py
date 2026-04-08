FACE_SWAP_UPSCALE_CHOICES = ("128px", "256px", "384px", "512px", "768px", "1024px")
DEFAULT_FACE_SWAP_MODEL = "inswapper_128"

FACE_SWAP_MODEL_SET = {
    "inswapper_128": {
        "filename": "inswapper_128.onnx",
        "url": "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
        "tile_size": 128,
        "type": "inswapper",
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "upscale_choices": ["128px", "256px", "384px", "512px", "768px", "1024px"],
    },
    "inswapper_128_fp16": {
        "filename": "inswapper_128_fp16.onnx",
        "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/inswapper_128_fp16.onnx",
        "tile_size": 128,
        "type": "inswapper",
        "template": "arcface_128",
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
        "upscale_choices": ["128px", "256px", "384px", "512px", "768px", "1024px"],
    },
    "hyperswap_1a_256": {
        "filename": "hyperswap_1a_256.onnx",
        "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
        "tile_size": 256,
        "type": "hyperswap",
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "upscale_choices": ["256px", "512px", "768px", "1024px"],
    },
    "hyperswap_1b_256": {
        "filename": "hyperswap_1b_256.onnx",
        "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx",
        "tile_size": 256,
        "type": "hyperswap",
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "upscale_choices": ["256px", "512px", "768px", "1024px"],
    },
    "hyperswap_1c_256": {
        "filename": "hyperswap_1c_256.onnx",
        "url": "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx",
        "tile_size": 256,
        "type": "hyperswap",
        "template": "arcface_128",
        "mean": [0.5, 0.5, 0.5],
        "standard_deviation": [0.5, 0.5, 0.5],
        "upscale_choices": ["256px", "512px", "768px", "1024px"],
    },
}


def _parse_upscale_size(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    digits = "".join(char for char in str(value) if char.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def parse_face_swap_upscale_size(value, model_name=None) -> int:
    return int(normalize_face_swap_upscale(value, model_name)[:-2])


def get_face_swap_model_key(model_name=None) -> str:
    candidate = str(model_name or "").strip()
    if candidate in FACE_SWAP_MODEL_SET:
        return candidate
    return DEFAULT_FACE_SWAP_MODEL


def get_face_swap_model_config(model_name=None) -> dict:
    return FACE_SWAP_MODEL_SET[get_face_swap_model_key(model_name)]


def get_face_swap_model_choices() -> list[str]:
    return list(FACE_SWAP_MODEL_SET.keys())


def get_face_swap_model_tile_size(model_name=None) -> int:
    return int(get_face_swap_model_config(model_name)["tile_size"])


def get_face_swap_model_type(model_name=None) -> str:
    return str(get_face_swap_model_config(model_name).get("type") or "inswapper")


def get_face_swap_model_template(model_name=None) -> str:
    return str(get_face_swap_model_config(model_name).get("template") or "arcface_128")


def get_face_swap_model_mean(model_name=None) -> list[float]:
    return list(get_face_swap_model_config(model_name).get("mean") or [0.0, 0.0, 0.0])


def get_face_swap_model_standard_deviation(model_name=None) -> list[float]:
    return list(get_face_swap_model_config(model_name).get("standard_deviation") or [1.0, 1.0, 1.0])


def get_face_swap_upscale_choices(model_name=None) -> list[str]:
    model_config = get_face_swap_model_config(model_name)
    configured_choices = model_config.get("upscale_choices")
    if configured_choices:
        return list(configured_choices)
    minimum_size = get_face_swap_model_tile_size(model_name)
    return [choice for choice in FACE_SWAP_UPSCALE_CHOICES if int(choice[:-2]) >= minimum_size]


def normalize_face_swap_upscale(value, model_name=None) -> str:
    requested_size = _parse_upscale_size(value)
    choices = get_face_swap_upscale_choices(model_name)
    minimum_size = int(choices[0][:-2])
    effective_size = max(requested_size or minimum_size, minimum_size)
    for choice in choices:
        if int(choice[:-2]) >= effective_size:
            return choice
    return choices[-1]


def coerce_face_swap_subsample_size(value, model_name=None) -> int:
    return parse_face_swap_upscale_size(value, model_name)


def get_face_swap_model_path(model_name=None) -> str:
    from roop.utils import resolve_relative_path

    return resolve_relative_path(
        f"../models/{get_face_swap_model_config(model_name)['filename']}"
    )


def ensure_face_swap_model_downloaded(model_name=None) -> str:
    from roop.utils import conditional_download, resolve_relative_path

    model_config = get_face_swap_model_config(model_name)
    conditional_download(resolve_relative_path("../models"), [model_config["url"]])
    return get_face_swap_model_path(model_name)


def get_face_swap_model_hint(model_name=None) -> str:
    model_key = get_face_swap_model_key(model_name)
    tile_size = get_face_swap_model_tile_size(model_key)
    upscale_choices = ", ".join(get_face_swap_upscale_choices(model_key))
    return (
        f"Selected `{model_key}`. Base swap tile: `{tile_size}x{tile_size}`. "
        f"`Subsample upscale` stays at or above `{tile_size}px` ({upscale_choices}). "
        "Models download automatically on first use."
    )


def get_face_swap_upscale_hint(model_name=None) -> str:
    tile_size = get_face_swap_model_tile_size(model_name)
    upscale_choices = ", ".join(get_face_swap_upscale_choices(model_name))
    return f"Model-aware pixel boost. Available for this model: {upscale_choices}. Minimum is {tile_size}px."
