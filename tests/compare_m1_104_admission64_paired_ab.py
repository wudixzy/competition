#!/usr/bin/env python3
"""Compare three fixed fine32/admission64 TP4 policy pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

try:
    from tests import bench_m1_104_admission64_policy_matrix as measurement
except ImportError:
    import bench_m1_104_admission64_policy_matrix as measurement


Json = dict[str, Any]
SCHEMA = "bi100-m1-104-admission64-paired-ab-v1"
VERSION = 1
PAIR_COUNT = 3
MIN_HIT = 0.50
MIN_POSITIVE_PAIRS = 2
MIN_HIT_GAIN = 0.02
MIN_WEIGHTED_GAIN = 0.03
MAX_MEDIAN_REGRESSION = 0.02
MAX_SINGLE_REGRESSION = 0.05


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} is below {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} exceeds {maximum}")
    return result


def _digest(value: Any, field: str) -> str:
    if not measurement.SHA256_RE.fullmatch(str(value or "")):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _request_contract(value: Any, field: str) -> tuple[Any, ...]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    request_id = value.get("request_id")
    target = value.get("target_prompt_tokens")
    pair = value.get("pair")
    phase = value.get("phase")
    rendered = value.get("rendered_tokens_local")
    seed = value.get("seed")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"{field}.request_id is invalid")
    if target not in measurement.SHAPES:
        raise ValueError(f"{field}.target_prompt_tokens is invalid")
    if pair not in measurement.PAIRS:
        raise ValueError(f"{field}.pair is invalid")
    if phase not in measurement.PHASES:
        raise ValueError(f"{field}.phase is invalid")
    if not isinstance(rendered, int) or isinstance(rendered, bool):
        raise ValueError(f"{field}.rendered_tokens_local is invalid")
    if seed != measurement.SEED:
        raise ValueError(f"{field}.seed differs")
    return (
        request_id,
        target,
        pair,
        phase,
        _digest(value.get("salt_sha256"), f"{field}.salt_sha256"),
        rendered,
        seed,
    )


def _output_identity(value: Json, field: str) -> tuple[Any, ...]:
    completion = value.get("completion_tokens")
    finish = value.get("finish_reason")
    if (
        not isinstance(completion, int)
        or isinstance(completion, bool)
        or not 0 < completion <= measurement.MAX_TOKENS
    ):
        raise ValueError(f"{field}.completion_tokens is invalid")
    if not isinstance(finish, str) or not finish:
        raise ValueError(f"{field}.finish_reason is invalid")
    return (
        _digest(value.get("first_token_sha256"),
                f"{field}.first_token_sha256"),
        _digest(value.get("output_sha256"), f"{field}.output_sha256"),
        _digest(value.get("content_sha256"), f"{field}.content_sha256"),
        _digest(value.get("reasoning_sha256"),
                f"{field}.reasoning_sha256"),
        _digest(value.get("tool_calls_sha256"),
                f"{field}.tool_calls_sha256"),
        finish,
        completion,
    )


def _validate_report(
    value: Json,
    *,
    mode: str,
    policy: str,
    field: str,
) -> tuple[list[Json], Json]:
    if (
        value.get("schema") != measurement.SCHEMA
        or value.get("version") != measurement.VERSION
        or value.get("mode") != mode
        or value.get("policy") != policy
        or value.get("request_count") != measurement.REQUEST_COUNT
        or value.get("qualified_measurement") is not True
        or value.get("reasons") != []
    ):
        raise ValueError(f"{field} measurement contract differs")
    fixed = value.get("fixed")
    if not isinstance(fixed, dict):
        raise ValueError(f"{field}.fixed must be an object")
    expected_fixed = {
        "shapes": list(measurement.SHAPES),
        "pairs": list(measurement.PAIRS),
        "phases": list(measurement.PHASES),
        "seed": measurement.SEED,
        "tool_count": measurement.TOOL_COUNT,
        "max_tokens": measurement.MAX_TOKENS,
        "temperature": 0,
        "thinking": False,
        "tool_choice": "none",
        "stream_usage": True,
    }
    for name, expected in expected_fixed.items():
        if fixed.get(name) != expected:
            raise ValueError(f"{field}.fixed.{name} differs")
    _digest(
        fixed.get("salt_namespace_sha256"),
        f"{field}.fixed.salt_namespace_sha256",
    )
    corpus = fixed.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise ValueError(f"{field}.fixed.corpus is invalid")
    for index, item in enumerate(corpus):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
        ):
            raise ValueError(f"{field}.fixed.corpus[{index}] is invalid")
        _digest(item.get("sha256"),
                f"{field}.fixed.corpus[{index}].sha256")
    if value.get("privacy") != {
        "contains_raw_prompt": False,
        "contains_raw_output": False,
        "contains_tools": False,
        "contains_credentials": False,
    }:
        raise ValueError(f"{field}.privacy differs")

    aggregate = value.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{field}.aggregate must be an object")
    _finite(
        aggregate.get("success_rate"),
        f"{field}.aggregate.success_rate",
        minimum=0.0,
        maximum=1.0,
    )
    _finite(
        aggregate.get("effective_hit_rate"),
        f"{field}.aggregate.effective_hit_rate",
        minimum=0.0,
        maximum=1.0,
    )
    for name in (
        "output_tps_p10",
        "input_tps",
        "cache_tps",
        "ttft_p90_s",
        "weighted",
        "prompt_tokens",
        "cached_tokens",
        "cold_cached_tokens",
        "first_request_cached_tokens",
    ):
        _finite(
            aggregate.get(name),
            f"{field}.aggregate.{name}",
            minimum=0.0,
        )

    requests = value.get("requests")
    if (
        not isinstance(requests, list)
        or len(requests) != measurement.REQUEST_COUNT
    ):
        raise ValueError(f"{field}.requests must contain 18 objects")
    contracts = [
        _request_contract(request, f"{field}.requests[{index}]")
        for index, request in enumerate(requests)
    ]
    request_reasons = measurement.validate_requests(requests)
    if request_reasons:
        raise ValueError(
            f"{field}.requests fail validation: {request_reasons[0]}")
    for index, request in enumerate(requests):
        if (
            request.get("ok") is not True
            or request.get("http_status") != 200
            or request.get("done_seen") is not True
            or request.get("health_after") is not True
        ):
            raise ValueError(f"{field}.requests[{index}] is not successful")
        _output_identity(request, f"{field}.requests[{index}]")
        for name in ("ttft_s", "latency_s", "decode_window_s", "output_tps"):
            _finite(
                request.get(name),
                f"{field}.requests[{index}].{name}",
                minimum=0.0,
            )
    target_order = value.get("target_order")
    if target_order != [contract[0] for contract in contracts]:
        raise ValueError(f"{field}.target_order differs")
    manifest = _digest(
        value.get("request_manifest_sha256"),
        f"{field}.request_manifest_sha256",
    )
    if manifest != measurement._sha256_json(contracts):
        raise ValueError(f"{field}.request_manifest_sha256 is not bound")
    recomputed = measurement.aggregate(requests)
    for name, expected in recomputed.items():
        observed = aggregate.get(name)
        if isinstance(expected, float):
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"{field}.aggregate.{name} is not bound")
        elif observed != expected:
            raise ValueError(f"{field}.aggregate.{name} is not bound")
    return requests, {
        "contracts": contracts,
        "manifest": manifest,
        "fixed": fixed,
        "aggregate": aggregate,
    }


def compare(
    controls: list[Json],
    candidates: list[Json],
    *,
    input_bindings: list[Json] | None = None,
) -> Json:
    reasons: list[str] = []
    pairs: list[Json] = []
    signatures: list[tuple[Any, ...]] = []
    if len(controls) != PAIR_COUNT or len(candidates) != PAIR_COUNT:
        reasons.append("exactly three control/candidate pairs are required")
    for pair_index, (control, candidate) in enumerate(
            zip(controls, candidates), 1):
        label = f"pair[{pair_index}]"
        try:
            control_requests, control_meta = _validate_report(
                control,
                mode="control",
                policy="fine32",
                field=f"{label}.control",
            )
            candidate_requests, candidate_meta = _validate_report(
                candidate,
                mode="candidate",
                policy="admission64",
                field=f"{label}.candidate",
            )
        except ValueError as error:
            reasons.append(str(error))
            continue

        signature = (
            control_meta["manifest"],
            tuple(control["target_order"]),
            json.dumps(control_meta["fixed"], sort_keys=True),
        )
        signatures.append(signature)
        if (
            control_meta["manifest"] != candidate_meta["manifest"]
            or control["target_order"] != candidate["target_order"]
            or control_meta["fixed"] != candidate_meta["fixed"]
            or control_meta["contracts"] != candidate_meta["contracts"]
        ):
            reasons.append(f"{label} request workload differs")

        exact_outputs = True
        for request_index, (control_request, candidate_request) in enumerate(
                zip(control_requests, candidate_requests)):
            try:
                if _output_identity(
                    control_request,
                    f"{label}.control.requests[{request_index}]",
                ) != _output_identity(
                    candidate_request,
                    f"{label}.candidate.requests[{request_index}]",
                ):
                    exact_outputs = False
                    reasons.append(
                        f"{label} request[{request_index}] output differs")
            except ValueError as error:
                exact_outputs = False
                reasons.append(str(error))

        control_aggregate = control_meta["aggregate"]
        candidate_aggregate = candidate_meta["aggregate"]
        control_output = float(control_aggregate["output_tps_p10"])
        candidate_output = float(candidate_aggregate["output_tps_p10"])
        control_ttft = float(control_aggregate["ttft_p90_s"])
        candidate_ttft = float(candidate_aggregate["ttft_p90_s"])
        control_weighted = float(control_aggregate["weighted"])
        candidate_weighted = float(candidate_aggregate["weighted"])
        control_hit = float(control_aggregate["effective_hit_rate"])
        candidate_hit = float(candidate_aggregate["effective_hit_rate"])
        if min(control_output, control_ttft, control_weighted) <= 0:
            reasons.append(f"{label} control denominator is non-positive")
            continue

        output_regression = candidate_output / control_output - 1.0
        ttft_regression = candidate_ttft / control_ttft - 1.0
        weighted_gain = candidate_weighted / control_weighted - 1.0
        hit_gain = candidate_hit - control_hit
        benefit_paths = {
            "effective_hit_gain_at_least_2pp":
                hit_gain + 1e-12 >= MIN_HIT_GAIN,
            "weighted_gain_at_least_3pct_without_hit_reduction": (
                weighted_gain + 1e-12 >= MIN_WEIGHTED_GAIN
                and hit_gain + 1e-12 >= 0.0
            ),
        }
        if candidate_hit + 1e-12 < MIN_HIT:
            reasons.append(f"{label} candidate effective hit is below 50%")
        if (
            control_aggregate["success_rate"] != 1.0
            or candidate_aggregate["success_rate"] != 1.0
        ):
            reasons.append(f"{label} success rate is below 100%")
        if (
            control_aggregate["first_request_cached_tokens"] != 0
            or candidate_aggregate["first_request_cached_tokens"] != 0
        ):
            reasons.append(
                f"{label} first request of a fresh service is not cold")
        if candidate_output + 1e-12 < 20.0:
            reasons.append(f"{label} candidate Output TPS P10 is below 20")
        if output_regression < -MAX_SINGLE_REGRESSION - 1e-12:
            reasons.append(
                f"{label} Output TPS regression exceeds 5%")
        if ttft_regression > MAX_SINGLE_REGRESSION + 1e-12:
            reasons.append(f"{label} TTFT P90 regression exceeds 5%")
        pairs.append({
            "pair": pair_index,
            "control_effective_hit_rate": control_hit,
            "candidate_effective_hit_rate": candidate_hit,
            "effective_hit_gain": hit_gain,
            "weighted_gain": weighted_gain,
            "output_tps_regression": output_regression,
            "ttft_p90_regression": ttft_regression,
            "benefit_paths": benefit_paths,
            "benefit_qualified": any(benefit_paths.values()),
            "exact_request_outputs": exact_outputs,
        })

    summary = {
        "positive_pairs": sum(
            pair["benefit_qualified"] for pair in pairs),
        "median_effective_hit_gain": (
            statistics.median([
                pair["effective_hit_gain"] for pair in pairs])
            if pairs else None
        ),
        "median_weighted_gain": (
            statistics.median([
                pair["weighted_gain"] for pair in pairs])
            if pairs else None
        ),
        "median_output_tps_regression": (
            statistics.median([
                pair["output_tps_regression"] for pair in pairs])
            if pairs else None
        ),
        "median_ttft_p90_regression": (
            statistics.median([
                pair["ttft_p90_regression"] for pair in pairs])
            if pairs else None
        ),
    }
    if len(pairs) == PAIR_COUNT:
        if len(set(signatures)) != 1:
            reasons.append(
                "all three pairs must use the same fixed workload")
        median_paths = {
            "effective_hit_gain_at_least_2pp": (
                summary["median_effective_hit_gain"] + 1e-12
                >= MIN_HIT_GAIN
            ),
            "weighted_gain_at_least_3pct_without_hit_reduction": (
                summary["median_weighted_gain"] + 1e-12
                >= MIN_WEIGHTED_GAIN
                and summary["median_effective_hit_gain"] + 1e-12 >= 0.0
            ),
        }
        summary["median_benefit_paths"] = median_paths
        if summary["positive_pairs"] < MIN_POSITIVE_PAIRS:
            reasons.append(
                "fewer than two pairs pass a policy-v2 benefit path")
        if not any(median_paths.values()):
            reasons.append(
                "median policy-v2 hit/weighted benefit is insufficient")
        if (
            summary["median_output_tps_regression"]
            < -MAX_MEDIAN_REGRESSION - 1e-12
        ):
            reasons.append("median Output TPS regression exceeds 2%")
        if (
            summary["median_ttft_p90_regression"]
            > MAX_MEDIAN_REGRESSION + 1e-12
        ):
            reasons.append("median TTFT P90 regression exceeds 2%")
    else:
        summary["median_benefit_paths"] = {}

    qualified = len(pairs) == PAIR_COUNT and not reasons
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "reasons": reasons,
        "pair_count": len(pairs),
        "thresholds": {
            "minimum_candidate_effective_hit_rate": MIN_HIT,
            "minimum_positive_pairs": MIN_POSITIVE_PAIRS,
            "minimum_effective_hit_gain": MIN_HIT_GAIN,
            "minimum_weighted_gain": MIN_WEIGHTED_GAIN,
            "maximum_median_regression": MAX_MEDIAN_REGRESSION,
            "maximum_single_regression": MAX_SINGLE_REGRESSION,
            "minimum_candidate_output_tps_p10": 20.0,
        },
        "pairs": pairs,
        "summary": summary,
        "inputs": input_bindings or [],
        "decision": {
            "m1_85_full_quality_authorized": qualified,
            "official_style_replay_authorized": False,
            "default_policy_change_authorized": False,
            "production_promotion_authorized": False,
            "yaml_change_authorized": False,
            "main_merge_authorized": False,
        },
        "official_881_evaluated": False,
        "privacy": {
            "contains_raw_prompt": False,
            "contains_raw_output": False,
            "contains_credentials": False,
        },
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="append", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bindings = []
    try:
        controls = [_load(path) for path in args.control]
        candidates = [_load(path) for path in args.candidate]
        for mode, paths in (
            ("control", args.control),
            ("candidate", args.candidate),
        ):
            bindings.extend({
                "mode": mode,
                "pair": pair,
                "path_name": path.name,
                "sha256": _sha256(path),
            } for pair, path in enumerate(paths, 1))
        report = compare(
            controls,
            candidates,
            input_bindings=bindings,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = compare([], [], input_bindings=bindings)
        report["reasons"].append(
            f"input loading failed: {type(error).__name__}")
        report["qualified"] = False
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
