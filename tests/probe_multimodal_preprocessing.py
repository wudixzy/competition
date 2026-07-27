#!/usr/bin/env python3
"""Exercise Qwen3.6 multi-image preprocessing without loading weights."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import struct
import zlib
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def _solid_png_data_url(rgb: tuple[int, int, int]) -> str:
    if len(rgb) != 3 or any(value < 0 or value > 255 for value in rgb):
        raise ValueError("invalid RGB value")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    width = height = 128
    scanline = b"\x00" + bytes(rgb) * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    image = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanline * height, level=9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def _digest_ints(values: list[int]) -> str:
    encoded = json.dumps(
        values, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest_tensor_tree(value: Any) -> str:
    import torch

    digest = hashlib.sha256(b"bi100-synthetic-tensor-tree-v1\0")

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            descriptor = (
                f"{tensor.dtype}:{tuple(tensor.shape)}:"
            ).encode("ascii")
            digest.update(b"T")
            digest.update(descriptor)
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        else:
            raise TypeError(f"unsupported mapped value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _tensor_tree_finite(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (list, tuple)):
        return all(_tensor_tree_finite(item) for item in value)
    return False


def _tensor_tree_shapes(value: Any) -> list[list[int]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, (list, tuple)):
        shapes = []
        for item in value:
            shapes.extend(_tensor_tree_shapes(item))
        return shapes
    raise TypeError(f"unsupported mapped value: {type(value).__name__}")


def _model_config(model_path: Path, image_limit: int):
    from vllm.config import ModelConfig

    return ModelConfig(
        model=str(model_path),
        tokenizer=str(model_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="half",
        seed=0,
        max_model_len=262144,
        skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": image_limit},
    )


def _payload(urls: list[str]) -> Json:
    content: list[Json] = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in urls
    ]
    content.append({"type": "text", "text": "describe synthetic images"})
    return {
        "model": "llm",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
    }


def _image_count(mm_data: Json | None) -> int:
    if not mm_data or "image" not in mm_data:
        return 0
    images = mm_data["image"]
    return len(images) if isinstance(images, list) else 1


def _process_case(model_config, tokenizer, urls: list[str]) -> Json:
    from vllm.entrypoints.chat_utils import (
        apply_hf_chat_template,
        parse_chat_messages,
    )
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    from vllm.inputs import INPUT_REGISTRY
    from vllm.multimodal import MULTIMODAL_REGISTRY

    request = ChatCompletionRequest.model_validate(_payload(urls))
    conversation, mm_data = parse_chat_messages(
        request.messages, model_config, tokenizer)
    prompt = apply_hf_chat_template(
        tokenizer,
        conversation,
        chat_template=None,
        add_generation_prompt=True,
    )
    prompt_token_ids = list(
        tokenizer.encode(prompt, add_special_tokens=False))
    image_token_id = int(model_config.hf_config.image_token_id)
    original_placeholders = prompt_token_ids.count(image_token_id)

    processed = INPUT_REGISTRY.process_input(model_config, {
        "prompt": prompt,
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": mm_data,
    })
    processed_ids = list(processed["prompt_token_ids"])
    mapped = MULTIMODAL_REGISTRY.map_input(model_config, mm_data or {})
    mapped_keys = sorted(mapped)
    mapped_shapes = {
        key: _tensor_tree_shapes(mapped[key])
        for key in mapped_keys
    }
    mapped_finite = {
        key: _tensor_tree_finite(mapped[key])
        for key in mapped_keys
    }
    mapped_sha256 = {
        key: _digest_tensor_tree(mapped[key])
        for key in mapped_keys
    }

    return {
        "image_count": _image_count(mm_data),
        "original_prompt_tokens": len(prompt_token_ids),
        "original_image_placeholders": original_placeholders,
        "processed_prompt_tokens": len(processed_ids),
        "processed_image_tokens": processed_ids.count(image_token_id),
        "processed_token_sha256": _digest_ints(processed_ids),
        "mapped_keys": mapped_keys,
        "mapped_shapes": mapped_shapes,
        "mapped_finite": mapped_finite,
        "mapped_sha256": mapped_sha256,
    }


def _default_two_image_rejection(model_path: Path, tokenizer,
                                 urls: list[str]) -> Json:
    from vllm.entrypoints.chat_utils import parse_chat_messages
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

    config = _model_config(model_path, 1)
    request = ChatCompletionRequest.model_validate(_payload(urls))
    try:
        parse_chat_messages(request.messages, config, tokenizer)
    except Exception as exc:
        message = str(exc)
        reason = (
            "image_count_limit"
            if (message.startswith("At most ")
                and " image(s) may be provided in one request." in message)
            else "unclassified"
        )
        return {
            "rejected": True,
            "reason": reason,
            "error_type": type(exc).__name__,
        }
    return {"rejected": False, "reason": None, "error_type": None}


def _package_versions() -> Json:
    versions: Json = {}
    for package in ("torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def probe(model_path: Path) -> Json:
    from transformers import AutoTokenizer
    from vllm.multimodal import MULTIMODAL_REGISTRY

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    config = _model_config(model_path, 2)
    MULTIMODAL_REGISTRY.init_mm_limits_per_prompt(config)

    red = _solid_png_data_url((255, 0, 0))
    blue = _solid_png_data_url((0, 0, 255))
    one = _process_case(config, tokenizer, [red])
    ordered = _process_case(config, tokenizer, [red, blue])
    repeated = _process_case(config, tokenizer, [red, blue])
    reversed_order = _process_case(config, tokenizer, [blue, red])
    default_rejection = _default_two_image_rejection(
        model_path, tokenizer, [red, blue])

    ordered_finite = all(ordered["mapped_finite"].values())
    checks = {
        "max_model_len_262144": config.max_model_len == 262144,
        "default_rejects_second_image": (
            default_rejection["rejected"]
            and default_rejection["reason"] == "image_count_limit"
        ),
        "explicit_two_accepts_two_images": (
            ordered["image_count"] == 2
            and ordered["original_image_placeholders"] == 2
        ),
        "visual_token_count_scales_linearly": (
            one["processed_image_tokens"] > 0
            and ordered["processed_image_tokens"]
            == 2 * one["processed_image_tokens"]
        ),
        "repeat_is_exact": ordered == repeated,
        "image_order_is_content_sensitive": (
            ordered["processed_token_sha256"]
            != reversed_order["processed_token_sha256"]
            and ordered["mapped_sha256"] != reversed_order["mapped_sha256"]
        ),
        "mapped_tensors_are_finite": ordered_finite,
        "mapped_contract_present": (
            "pixel_values" in ordered["mapped_keys"]
            and "image_grid_thw" in ordered["mapped_keys"]
        ),
    }
    return {
        "schema": "bi100-multimodal-preprocessing-probe-v1",
        "synthetic_only": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "model": {
            "model_type": getattr(config.hf_config, "model_type", None),
            "architectures":
                list(getattr(config.hf_config, "architectures", ()) or ()),
            "max_model_len": config.max_model_len,
        },
        "default_limit": default_rejection,
        "one_image": one,
        "two_images": ordered,
        "two_images_repeated": repeated,
        "two_images_reversed": reversed_order,
        "package_versions": _package_versions(),
        "privacy": {
            "contains_prompt_or_response_text": False,
            "contains_image_url_or_bytes": False,
            "contains_model_weights": False,
            "synthetic_image_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    report = probe(args.model_path)
    rendered = json.dumps(
        report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
