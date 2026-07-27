#!/usr/bin/env python3
"""Probe CoreX vLLM multimodal limits without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]


def _digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_config(model_path: Path, image_limit: int | None):
    from vllm.config import ModelConfig

    kwargs: Json = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "tokenizer_mode": "auto",
        "trust_remote_code": True,
        "dtype": "half",
        "seed": 0,
        "max_model_len": 262144,
        "skip_tokenizer_init": True,
    }
    if image_limit is not None:
        kwargs["limit_mm_per_prompt"] = {"image": image_limit}
    return ModelConfig(**kwargs)


def _tracker_observation(model_config, image_count: int) -> Json:
    from vllm.entrypoints.chat_utils import MultiModalItemTracker

    class _Tokenizer:

        @staticmethod
        def decode(_token_index: int) -> str:
            return "synthetic"

    tracker = MultiModalItemTracker(model_config, _Tokenizer())
    accepted = 0
    reason = None
    error_type = None
    for _ in range(image_count):
        try:
            tracker.add("image", {"image": object()})
        except Exception as exc:
            error_type = type(exc).__name__
            message = str(exc)
            if (message.startswith("At most ")
                    and " image(s) may be provided in one request." in message):
                reason = "image_count_limit"
            else:
                reason = "unclassified"
            break
        accepted += 1

    combined = tracker.all_mm_data()
    combined_images = None if combined is None else combined.get("image")
    if isinstance(combined_images, list):
        combined_count = len(combined_images)
    elif combined_images is None:
        combined_count = 0
    else:
        combined_count = 1
    return {
        "attempted": image_count,
        "accepted": accepted,
        "combined_count": combined_count,
        "reason": reason,
        "error_type": error_type,
    }


def _package_versions() -> Json:
    versions: Json = {}
    for package in ("torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def probe(model_path: Path) -> Json:
    from vllm.multimodal import MULTIMODAL_REGISTRY

    default = _model_config(model_path, None)
    MULTIMODAL_REGISTRY.init_mm_limits_per_prompt(default)
    default_effective = dict(
        MULTIMODAL_REGISTRY.get_mm_limits_per_prompt(default))
    default_tokens = MULTIMODAL_REGISTRY.get_max_multimodal_tokens(default)

    explicit_two = _model_config(model_path, 2)
    MULTIMODAL_REGISTRY.init_mm_limits_per_prompt(explicit_two)
    explicit_effective = dict(
        MULTIMODAL_REGISTRY.get_mm_limits_per_prompt(explicit_two))
    explicit_tokens = MULTIMODAL_REGISTRY.get_max_multimodal_tokens(
        explicit_two)

    default_tracker = _tracker_observation(default, 2)
    explicit_tracker = _tracker_observation(explicit_two, 2)
    architecture = list(getattr(default.hf_config, "architectures", ()) or ())
    model_type = getattr(default.hf_config, "model_type", None)

    checks = {
        "qwen36_model_identity": (
            model_type == "qwen3_5_moe"
            and "Qwen3_5MoeForCausalLM" in architecture
        ),
        "max_model_len_262144": default.max_model_len == 262144,
        "default_effective_image_limit_one":
            default_effective.get("image") == 1,
        "explicit_effective_image_limit_two":
            explicit_effective.get("image") == 2,
        "profile_budget_scales_linearly":
            default_tokens > 0 and explicit_tokens == 2 * default_tokens,
        "default_tracker_rejects_second_image": (
            default_tracker["accepted"] == 1
            and default_tracker["reason"] == "image_count_limit"
        ),
        "explicit_tracker_accepts_two_images": (
            explicit_tracker["accepted"] == 2
            and explicit_tracker["combined_count"] == 2
            and explicit_tracker["reason"] is None
        ),
    }
    return {
        "schema": "bi100-multimodal-limit-probe-v1",
        "synthetic_only": True,
        "qualified": all(checks.values()),
        "checks": checks,
        "model": {
            "model_type": model_type,
            "architectures": architecture,
            "max_model_len": default.max_model_len,
            "config_sha256": _digest(model_path / "config.json"),
            "tokenizer_config_sha256":
                _digest(model_path / "tokenizer_config.json"),
        },
        "default": {
            "configured": dict(default.multimodal_config.limit_per_prompt),
            "effective": default_effective,
            "max_multimodal_tokens": default_tokens,
            "tracker": default_tracker,
        },
        "explicit_image_two": {
            "configured":
                dict(explicit_two.multimodal_config.limit_per_prompt),
            "effective": explicit_effective,
            "max_multimodal_tokens": explicit_tokens,
            "tracker": explicit_tracker,
        },
        "package_versions": _package_versions(),
        "privacy": {
            "contains_prompt_or_response_text": False,
            "contains_image_url_or_bytes": False,
            "contains_model_weights": False,
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
