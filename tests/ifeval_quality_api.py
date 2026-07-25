#!/usr/bin/env python3
"""Run the frozen Google IFEval subset against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

import quality_runtime_contract as runtime_contract


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = ROOT / "quality/external/google_ifeval"
DEFAULT_MANIFEST = EXTERNAL_ROOT / "manifest.v1.json"
EXPECTED_MANIFEST_SHA256 = (
    "07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855"
)
REPORT_SCHEMA = "bi100-ifeval-result-v1"
REPORT_VERSION = 1
Json = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, value: Json, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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


def load_manifest(path: Path) -> tuple[Json, str, list[Json]]:
    digest = sha256(path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError("canonical IFEval manifest SHA-256 differs")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "bi100-ifeval-manifest-v1"
            or manifest.get("version") != 1
            or manifest.get("selection", {}).get("size") != 64
            or manifest.get("subset", {}).get("rows") != 64):
        raise ValueError("canonical IFEval manifest is invalid")
    subset = ROOT / manifest["subset"]["repository_path"]
    if (not subset.is_file()
            or subset.stat().st_size != manifest["subset"]["bytes"]
            or sha256(subset) != manifest["subset"]["sha256"]):
        raise ValueError("canonical IFEval subset identity differs")
    rows = [json.loads(line) for line in subset.read_text(
        encoding="utf-8").splitlines()]
    expected_keys = manifest["selection"]["selected_keys_in_request_order"]
    if [row.get("key") for row in rows] != expected_keys:
        raise ValueError("canonical IFEval request order differs")
    return manifest, digest, rows


def request_payload(prompt: str, model: str, manifest: Json) -> Json:
    contract = manifest["request_conversion"]
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": contract["max_tokens"],
        "temperature": contract["temperature"],
        "seed": contract["seed"],
        "stream": contract["stream"],
    }


def post(base: str, payload: Json, timeout_s: float) -> tuple[Json, float]:
    request = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return (json.loads(response.read().decode("utf-8")),
                    time.monotonic() - started)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"http_{exc.code}:body_sha256={hashlib.sha256(body).hexdigest()}"
        ) from exc


def normalize_response(body: Json, elapsed_s: float) -> Json:
    if not isinstance(body, dict):
        raise ValueError("response root must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise ValueError("response message is missing")
    content = message.get("content")
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(content, str) or not content:
        raise ValueError("response content must be nonempty text")
    if not isinstance(reasoning, str):
        raise ValueError("reasoning_content must be text when present")
    if message.get("tool_calls") not in (None, []):
        raise ValueError("IFEval response unexpectedly contains tool calls")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in ("stop", "length"):
        raise ValueError("IFEval finish reason is invalid")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("IFEval usage is missing")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        if (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"IFEval usage field is invalid: {name}")
    if usage["total_tokens"] != (
            usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ValueError("IFEval usage total is inconsistent")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    if not isinstance(cached, int) or isinstance(cached, bool) or cached < 0:
        raise ValueError("IFEval cached token count is invalid")
    return {
        "content": content,
        "reasoning_content": reasoning,
        "elapsed_s": elapsed_s,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_tokens": cached,
        },
    }


def score_rows(rows: list[Json], responses: dict[int, str]) -> list[Json]:
    sys.path.insert(0, str(EXTERNAL_ROOT))
    try:
        import langdetect  # type: ignore
        from instruction_following_eval import evaluation_lib  # type: ignore
    finally:
        sys.path.pop(0)
    langdetect.DetectorFactory.seed = 0
    result = []
    for row in rows:
        inp = evaluation_lib.InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        prompt_to_response = {row["prompt"]: responses[row["key"]]}
        strict = evaluation_lib.test_instruction_following_strict(
            inp, prompt_to_response)
        loose = evaluation_lib.test_instruction_following_loose(
            inp, prompt_to_response)
        result.append({
            "key": row["key"],
            "instruction_id_list": list(row["instruction_id_list"]),
            "strict": list(strict.follow_instruction_list),
            "loose": list(loose.follow_instruction_list),
        })
    return result


def summarize(scores: list[Json]) -> Json:
    by_id: dict[str, Json] = collections.defaultdict(
        lambda: {"total": 0, "strict_passed": 0, "loose_passed": 0})
    by_family: dict[str, Json] = collections.defaultdict(
        lambda: {"total": 0, "strict_passed": 0, "loose_passed": 0})
    strict_prompts = 0
    loose_prompts = 0
    strict_instructions = 0
    loose_instructions = 0
    instruction_total = 0
    for row in scores:
        strict_prompts += all(row["strict"])
        loose_prompts += all(row["loose"])
        strict_instructions += sum(row["strict"])
        loose_instructions += sum(row["loose"])
        instruction_total += len(row["instruction_id_list"])
        for instruction_id, strict, loose in zip(
                row["instruction_id_list"], row["strict"], row["loose"]):
            family = instruction_id.split(":", 1)[0]
            for target in (by_id[instruction_id], by_family[family]):
                target["total"] += 1
                target["strict_passed"] += strict
                target["loose_passed"] += loose
    return {
        "prompt_total": len(scores),
        "strict_prompt_passed": strict_prompts,
        "strict_prompt_accuracy": strict_prompts / len(scores),
        "loose_prompt_passed": loose_prompts,
        "loose_prompt_accuracy": loose_prompts / len(scores),
        "instruction_total": instruction_total,
        "strict_instruction_passed": strict_instructions,
        "strict_instruction_accuracy": strict_instructions / instruction_total,
        "loose_instruction_passed": loose_instructions,
        "loose_instruction_accuracy": loose_instructions / instruction_total,
        "by_instruction_id": dict(sorted(by_id.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def parse_checkpoint(path: Path, run_id: str) -> dict[int, Json]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "bi100-ifeval-private-checkpoint-v1"
            or value.get("run_id_sha256") != run_id
            or value.get("contains_raw_model_outputs") is not True):
        raise ValueError("IFEval checkpoint identity differs")
    return {int(key): item for key, item in value["responses"].items()}


def write_checkpoint(path: Path, run_id: str, responses: dict[int, Json]) -> None:
    atomic_write(path, {
        "schema": "bi100-ifeval-private-checkpoint-v1",
        "version": 1,
        "run_id_sha256": run_id,
        "contains_raw_model_outputs": True,
        "must_remain_outside_repository": True,
        "responses": {str(key): value for key, value in responses.items()},
    }, mode=0o600)


def write_progress(
    path: Path,
    run_id: str,
    selected: int,
    responses: dict[int, Json],
    failures: dict[int, Json],
    last_ordinal: int,
    report_sha256: str | None = None,
) -> None:
    atomic_write(path, {
        "schema": "bi100-ifeval-progress-v1",
        "version": 1,
        "run_id_sha256": run_id,
        "selected": selected,
        "attempted": len(responses) + len(failures),
        "successful": len(responses),
        "errors": len(failures),
        "last_ordinal": last_ordinal,
        "complete": len(responses) + len(failures) == selected,
        "report_sha256": report_sha256,
        "failures": [
            {"key": key, **failures[key]} for key in sorted(failures)
        ],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
        },
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--explicit-key", type=int, action="append", default=[])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--runtime-overlay-sha256", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--gdn-cache-policy", choices=("fine32", "admission64"),
                        required=True)
    parser.add_argument("--gdn-restore-mode", choices=("direct", "aligned"),
                        required=True)
    parser.add_argument("--fused-prefill", choices=(0, 1), type=int,
                        required=True)
    parser.add_argument("--kv-eviction-policy", choices=("lru", "frequency"),
                        required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"report already exists: {args.out}")
    checkpoint = args.checkpoint.resolve()
    progress = args.progress.resolve()
    if checkpoint == ROOT or ROOT in checkpoint.parents:
        raise ValueError("raw IFEval checkpoint must remain outside repository")
    if not str(checkpoint).startswith("/tmp/"):
        raise ValueError("raw IFEval checkpoint must use a private /tmp path")
    if (progress == ROOT or ROOT in progress.parents
            or not str(progress).startswith("/tmp/")):
        raise ValueError("IFEval progress must remain under /tmp")
    if args.timeout_s <= 0:
        raise ValueError("timeout must be positive")
    manifest, manifest_sha, all_rows = load_manifest(args.manifest)
    explicit = list(dict.fromkeys(args.explicit_key))
    known = {row["key"] for row in all_rows}
    if any(key not in known for key in explicit):
        raise ValueError("explicit IFEval key is not in the frozen subset")
    rows = ([row for row in all_rows if row["key"] in set(explicit)]
            if explicit else all_rows)
    contract, runtime_contract_sha = runtime_contract.load_runtime_contract(
        args.runtime_contract,
        {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": args.runtime_overlay_sha256,
            "instance": args.instance,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "served_model_name": args.model,
        },
        require_cache_trace=True,
    )
    expected_optimization_environment = {
        "BI100_GDN_CACHE_POLICY": args.gdn_cache_policy,
        "BI100_GDN_RESTORE_MODE": args.gdn_restore_mode,
        "BI100_ATTN_COREX_FUSED_PREFILL": str(args.fused_prefill),
        "BI100_KV_EVICTION_POLICY": args.kv_eviction_policy,
    }
    for name, expected in expected_optimization_environment.items():
        if contract["environment"].get(name) != expected:
            raise ValueError(f"runtime contract optimization differs: {name}")
    run_contract = {
        "manifest_sha256": manifest_sha,
        "selected_keys": [row["key"] for row in rows],
        "request_conversion": manifest["request_conversion"],
        "model": args.model,
        "source_revision": args.source_revision,
        "runtime_overlay_sha256": args.runtime_overlay_sha256,
        "runtime_contract_sha256": runtime_contract_sha,
        "optimization": {
            "gdn_cache_policy": args.gdn_cache_policy,
            "gdn_restore_mode": args.gdn_restore_mode,
            "fused_prefill": bool(args.fused_prefill),
            "kv_eviction_policy": args.kv_eviction_policy,
        },
    }
    run_id = canonical_sha256(run_contract)
    responses = parse_checkpoint(checkpoint, run_id)
    failures: dict[int, Json] = {}
    resumed_ordinal = max((
        ordinal for ordinal, row in enumerate(rows, 1)
        if row["key"] in responses
    ), default=0)
    write_progress(
        progress, run_id, len(rows), responses, failures, resumed_ordinal)

    for ordinal, row in enumerate(rows, 1):
        key = row["key"]
        if key in responses:
            continue
        try:
            body, elapsed = post(
                args.base,
                request_payload(row["prompt"], args.model, manifest),
                args.timeout_s,
            )
            normalized = normalize_response(body, elapsed)
            responses[key] = normalized
            write_checkpoint(checkpoint, run_id, responses)
            write_progress(
                progress, run_id, len(rows), responses, failures, ordinal)
            print(f"[{ordinal}/{len(rows)}] key={key} ok", flush=True)
        except Exception as exc:  # The report records type and digest only.
            message = f"{type(exc).__name__}:{exc}"
            failures[key] = {
                "error_type": type(exc).__name__,
                "error_sha256": hashlib.sha256(
                    message.encode("utf-8")).hexdigest(),
            }
            write_progress(
                progress, run_id, len(rows), responses, failures, ordinal)
            print(f"[{ordinal}/{len(rows)}] key={key} failed", flush=True)

    scored_rows = [row for row in rows if row["key"] in responses]
    scores = score_rows(
        scored_rows,
        {key: value["content"] for key, value in responses.items()},
    ) if scored_rows else []
    score_by_key = {row["key"]: row for row in scores}
    complete = len(responses) == len(rows) and not failures
    full_selection = not explicit and len(rows) == 64
    cases = []
    for row in rows:
        key = row["key"]
        if key in failures:
            cases.append({
                "key": key,
                "instruction_id_list": row["instruction_id_list"],
                "status": "error",
                **failures[key],
            })
            continue
        response = responses[key]
        score = score_by_key[key]
        cases.append({
            "key": key,
            "instruction_id_list": row["instruction_id_list"],
            "status": "pass",
            "strict": score["strict"],
            "loose": score["loose"],
            "finish_reason": response["finish_reason"],
            "elapsed_s": response["elapsed_s"],
            "content_chars": len(response["content"]),
            "reasoning_chars": len(response["reasoning_content"]),
            "semantic_output_sha256": canonical_sha256({
                "content": response["content"],
                "reasoning_content": response["reasoning_content"],
                "finish_reason": response["finish_reason"],
            }),
            "usage": response["usage"],
        })
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id_sha256": run_id,
        "qualified": complete,
        "quality_run_eligible_for_baseline": complete and full_selection,
        "promotion_authorized": False,
        "manifest": {
            "path_name": args.manifest.name,
            "sha256": manifest_sha,
            "subset_sha256": manifest["subset"]["sha256"],
            "selected_keys": [row["key"] for row in rows],
            "full_selection": full_selection,
        },
        "runtime": {
            "source_revision": args.source_revision,
            "runtime_identity": args.runtime_identity,
            "runtime_overlay_sha256": args.runtime_overlay_sha256,
            "runtime_contract_sha256": runtime_contract_sha,
            "instance": args.instance,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "optimization": run_contract["optimization"],
        },
        "runtime_contract": {
            "sha256": runtime_contract_sha,
            "file_sha256": sha256(args.runtime_contract),
            "contract": contract,
        },
        "request_conversion": manifest["request_conversion"],
        "evaluator": {
            "revision": manifest["evaluator"]["revision"],
            "strict_and_loose_rules_unmodified": True,
            "language_detector_seed": 0,
        },
        "summary": summarize(scores) if scores else None,
        "transport": {
            "selected": len(rows),
            "completed": len(responses),
            "errors": len(failures),
        },
        "cases": cases,
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
            "checkpoint_deleted": True,
        },
    }
    atomic_write(args.out, report)
    checkpoint.unlink(missing_ok=True)
    write_progress(
        progress, run_id, len(rows), responses, failures, len(rows),
        report_sha256=sha256(args.out),
    )
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["qualified"],
        "baseline_eligible": report["quality_run_eligible_for_baseline"],
        "report_sha256": sha256(args.out),
    }, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
