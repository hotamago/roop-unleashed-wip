import roop.globals

from roop.onnx_batch import ensure_native_batch_model


NATIVE_BATCH_PATCH_PROCESSORS = {"faceswap"}
TENSORRT_SAFE_PROCESSORS = {"faceswap"}


def _provider_name(provider) -> str:
    if isinstance(provider, tuple):
        return str(provider[0])
    return str(provider)


def resolve_model_path_for_processor(model_path: str, processor_name: str) -> str:
    if processor_name in NATIVE_BATCH_PATCH_PROCESSORS:
        return ensure_native_batch_model(model_path)
    return model_path


def get_execution_providers_for_processor(processor_name: str):
    providers = list(getattr(roop.globals, "execution_providers", []) or [])
    if not providers:
        return providers
    if processor_name in TENSORRT_SAFE_PROCESSORS:
        return providers
    if any(_provider_name(provider) == "TensorrtExecutionProvider" for provider in providers):
        filtered_providers = [
            provider for provider in providers
            if _provider_name(provider) != "TensorrtExecutionProvider"
        ]
        if filtered_providers:
            return filtered_providers
    return providers
