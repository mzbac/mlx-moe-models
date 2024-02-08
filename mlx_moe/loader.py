import glob
import json
import logging
from pathlib import Path
from typing import Tuple, Union
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import get_model_path
from mlx_lm.tuner.utils import apply_lora_layers

from transformers import  AutoTokenizer, PreTrainedTokenizer

from . import phi2moe, qwen2moe

# Constants
MODEL_MAPPING = {
    "phi2moe": phi2moe,
    "qwen2moe": qwen2moe,
}


def _get_classes(config: dict):
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def load_moe(model_path: Union[str, Path],tokenizer_config={}, adapter_file: str = None) -> Tuple[nn.Module, PreTrainedTokenizer]:
    if isinstance(model_path, str):
        model_path = Path(model_path)

    model_path = get_model_path(model_path)
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
            quantization = config.get("quantization", None)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)
    if hasattr(model_class, "sanitize"):
        weights = model_class.sanitize(weights)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0]
            != config[
                "num_local_experts"
            ],  # avoid quantizing gate layers, otherwise we may get loss nan during fine-tuning
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    if adapter_file is not None:
        model = apply_lora_layers(model, adapter_file)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_config)

    model.eval()
    return model, tokenizer
