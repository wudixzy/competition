#!/usr/bin/env python3
"""Freeze a deterministic, instruction-stratified Google IFEval subset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "quality/external/google_ifeval"
SOURCE_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
SOURCE_SHA256 = (
    "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"
)
SOURCE_BYTES = 207111
SOURCE_ROWS = 541
EVALUATOR_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
SEED = "bi100-ifeval-v1-seed-20260725"
SUBSET_SIZE = 64
MIN_INSTRUCTION_COVERAGE = 4
EXPECTED_INSTRUCTION_IDS = {
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
}
VENDORED_FILES = {
    "LICENSE.google-research": (
        11358,
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    "UPSTREAM_EVALUATOR_README.md": (
        1457,
        "fef3c44408e4b39837af345095ccce7da798853fd60ea94a1de63afa9d3d67b3",
    ),
    "source/HUGGINGFACE_DATASET_CARD.md": (
        5523,
        "c3e2d7da8286ae27649f526e3153cc72f071c21a6a7702d06f09cef5f9c114f1",
    ),
    "instruction_following_eval/evaluation_lib.py": (
        6984,
        "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    ),
    "instruction_following_eval/instructions.py": (
        55162,
        "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    ),
    "instruction_following_eval/instructions_registry.py": (
        7240,
        "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    ),
    "instruction_following_eval/instructions_util.py": (
        19538,
        "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    ),
    "instruction_following_eval/requirements.upstream.txt": (
        35,
        "1b1716c9ac21b9cc2bd0d7354ed8f7b6982ccf35598529601a7c08508acec94a",
    ),
}
WHEELS = {
    "absl_py-2.5.0-py3-none-any.whl": (
        137410,
        "0f17b89f2a4eaaedc4f28c622998aa690564b3012a396a4ffad0821007fe03ba",
    ),
    "click-8.4.2-py3-none-any.whl": (
        119243,
        "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76",
    ),
    "defusedxml-0.7.1-py2.py3-none-any.whl": (
        25604,
        "a352e7e428770286cc899e2542b6cdaedb2b4953ff269a210103ec58f6198a61",
    ),
    "immutabledict-4.3.1-py3-none-any.whl": (
        5000,
        "c9facdc0ff30fdb8e35bd16532026cac472a549e182c94fa201b51b25e4bf7bf",
    ),
    "joblib-1.5.3-py3-none-any.whl": (
        309071,
        "5fc3c5039fc5ca8c0276333a188bbd59d6b7ab37fe6632daa76bc7f9ec18e713",
    ),
    "langdetect-1.0.9.tar.gz": (
        981474,
        "cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0",
    ),
    "nltk-3.10.0-py3-none-any.whl": (
        1716144,
        "54ff84d4916d3ef127e8953bee0023f6a6b320b75d634a19e06ef056d3d244bf",
    ),
    "regex-2026.7.19-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": (
        785515,
        "4458124d71339f505bf1fb94f69fd1bb8fa9d2481eebfef27c10ef4f2b9e12f6",
    ),
    "six-1.17.0-py2.py3-none-any.whl": (
        11050,
        "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    ),
    "tqdm-4.69.1-py3-none-any.whl": (
        675452,
        "0a654b96f7a2660cceb615b56f307ec2bef96c515409014a429a561981ab52b4",
    ),
}
SECRET_MARKERS = (
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "github_pat_",
    "ghp_",
    "MODELHUB_ACCESS_TOKEN",
)
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
Json = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")


def load_source(path: Path) -> list[Json]:
    if path.stat().st_size != SOURCE_BYTES or sha256(path) != SOURCE_SHA256:
        raise ValueError("IFEval source identity differs")
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines()]
    if len(rows) != SOURCE_ROWS:
        raise ValueError("IFEval source row count differs")
    keys = []
    observed_ids = set()
    for row in rows:
        if (not isinstance(row, dict)
                or set(row) != {"key", "prompt", "instruction_id_list", "kwargs"}
                or not isinstance(row["key"], int)
                or not isinstance(row["prompt"], str)
                or not row["prompt"]
                or not isinstance(row["instruction_id_list"], list)
                or not isinstance(row["kwargs"], list)
                or len(row["instruction_id_list"]) != len(row["kwargs"])):
            raise ValueError("IFEval source row schema differs")
        keys.append(row["key"])
        observed_ids.update(row["instruction_id_list"])
    if len(set(keys)) != len(keys):
        raise ValueError("IFEval keys are not unique")
    if observed_ids != EXPECTED_INSTRUCTION_IDS:
        raise ValueError("IFEval instruction registry differs")
    serialized = path.read_text(encoding="utf-8")
    if any(marker.lower() in serialized.lower() for marker in SECRET_MARKERS):
        raise ValueError("IFEval source contains a credential marker")
    return rows


def selection_rank(row: Json) -> str:
    row_sha = hashlib.sha256(canonical_bytes(row)).hexdigest()
    return hashlib.sha256(f"{SEED}\0{row_sha}".encode("ascii")).hexdigest()


def select_rows(rows: list[Json]) -> list[Json]:
    counts: collections.Counter[str] = collections.Counter()
    remaining = list(enumerate(rows))
    selected: list[tuple[int, Json]] = []

    while any(counts[item] < MIN_INSTRUCTION_COVERAGE
              for item in EXPECTED_INSTRUCTION_IDS):
        def candidate_rank(candidate: tuple[int, Json]) -> tuple[Any, ...]:
            index, row = candidate
            unique_ids = set(row["instruction_id_list"])
            gain = sum(counts[item] < MIN_INSTRUCTION_COVERAGE
                       for item in unique_ids)
            deficit = sum(max(MIN_INSTRUCTION_COVERAGE - counts[item], 0)
                          for item in unique_ids)
            return (-gain, -deficit, selection_rank(row), index)

        winner = min(remaining, key=candidate_rank)
        if candidate_rank(winner)[0] == 0:
            raise ValueError("unable to satisfy IFEval instruction coverage")
        remaining.remove(winner)
        selected.append(winner)
        counts.update(winner[1]["instruction_id_list"])
        if len(selected) > SUBSET_SIZE:
            raise ValueError("IFEval coverage exceeds subset size")

    for candidate in sorted(
            remaining, key=lambda item: (selection_rank(item[1]), item[0])):
        if len(selected) == SUBSET_SIZE:
            break
        selected.append(candidate)
        counts.update(candidate[1]["instruction_id_list"])

    if (len(selected) != SUBSET_SIZE
            or set(counts) != EXPECTED_INSTRUCTION_IDS
            or min(counts.values()) < MIN_INSTRUCTION_COVERAGE):
        raise ValueError("IFEval frozen subset coverage is invalid")
    return [row for _, row in sorted(selected)]


def verified_files(root: Path, expected: dict[str, tuple[int, str]]) -> list[Json]:
    result = []
    for name, (size, digest) in sorted(expected.items()):
        path = root / name
        if (not path.is_file() or path.stat().st_size != size
                or sha256(path) != digest):
            raise ValueError(f"vendored file identity differs: {name}")
        result.append({"path": name, "bytes": size, "sha256": digest})
    return result


def write_subset(path: Path, rows: list[Json]) -> None:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_manifest(
    external_root: Path,
    source: Path,
    subset: Path,
    rows: list[Json],
    downloaded_at_utc: str,
) -> Json:
    counts = collections.Counter(
        item for row in rows for item in row["instruction_id_list"])
    shape_counts = collections.Counter(
        len(row["instruction_id_list"]) for row in rows)
    script = Path(__file__).resolve()
    return {
        "schema": "bi100-ifeval-manifest-v1",
        "version": 1,
        "name": "google-ifeval-bi100-stratified-64-v1",
        "source": {
            "dataset": "google/IFEval",
            "author_or_org": "Google Research",
            "url": "https://huggingface.co/datasets/google/IFEval",
            "revision": SOURCE_REVISION,
            "split": "train",
            "downloaded_at_utc": downloaded_at_utc,
            "repository_path": str(source.relative_to(ROOT)),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "rows": SOURCE_ROWS,
            "license": "Apache-2.0",
            "redistribution_allowed": True,
        },
        "selection": {
            "algorithm": (
                "greedy maximum uncovered instruction IDs, then maximum "
                "remaining deficit, then ascending SHA-256(seed,row); fill "
                "by ascending SHA-256 and emit in source order"
            ),
            "seed": SEED,
            "size": SUBSET_SIZE,
            "minimum_per_instruction_id": MIN_INSTRUCTION_COVERAGE,
            "instruction_id_count": len(counts),
            "instruction_counts": dict(sorted(counts.items())),
            "instruction_arity_counts": {
                str(key): value for key, value in sorted(shape_counts.items())
            },
            "selected_keys_in_request_order": [row["key"] for row in rows],
        },
        "subset": {
            "repository_path": str(subset.relative_to(ROOT)),
            "bytes": subset.stat().st_size,
            "sha256": sha256(subset),
            "rows": len(rows),
            "stable_order": True,
        },
        "evaluator": {
            "implementation": "Google Research instruction_following_eval",
            "url": (
                "https://github.com/google-research/google-research/tree/"
                f"{EVALUATOR_REVISION}/instruction_following_eval"
            ),
            "revision": EVALUATOR_REVISION,
            "license": "Apache-2.0",
            "vendored_files": verified_files(external_root, VENDORED_FILES),
            "strict_and_loose_rules_unmodified": True,
            "dataset_difference_from_evaluator_repo": {
                "different_row_count": 1,
                "different_key": 2785,
                "selected": any(row["key"] == 2785 for row in rows),
                "evaluator_repo_data_sha256": (
                    "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
                ),
            },
        },
        "offline_environment": {
            "python": "CPython 3.10 x86_64",
            "distribution_artifacts": verified_files(
                external_root / "wheelhouse", WHEELS),
            "nltk_punkt_tab": {
                "repository_snapshot": False,
                "redistribution_allowed": False,
                "license": "not stated in pinned nltk_data index",
                "source_url": (
                    "https://raw.githubusercontent.com/nltk/nltk_data/"
                    "4f15a3d89eefe9748ec1c05be495d91289197155/"
                    "packages/tokenizers/punkt_tab.zip"
                ),
                "revision": "4f15a3d89eefe9748ec1c05be495d91289197155",
                "archive_sha256": (
                    "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
                ),
                "prestage_required": True,
            },
        },
        "request_conversion": {
            "endpoint": "/v1/chat/completions",
            "messages": "one user message containing the source prompt verbatim",
            "stream": False,
            "temperature": 0,
            "seed": 20260725,
            "max_tokens": 4096,
            "request_order": "selected_keys_in_request_order",
            "chat_template_override": False,
            "thinking_override": False,
        },
        "scoring": {
            "response_field": "choices[0].message.content",
            "strict": "official test_instruction_following_strict",
            "loose": "official test_instruction_following_loose",
            "baseline_rule": "complete transport-valid run establishes counts",
            "candidate_rule": (
                "no decrease in strict or loose prompt, instruction, or "
                "per-instruction-ID pass counts"
            ),
        },
        "generator": {
            "path": str(script.relative_to(ROOT)),
            "sha256": sha256(script),
        },
        "privacy": {
            "contains_credentials": False,
            "contains_private_user_data": False,
            "contains_official_competition_requests": False,
            "result_reports_may_contain_raw_prompts_or_outputs": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=DEFAULT_ROOT / "source/ifeval_input_data.jsonl")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_ROOT / "subset.v1.jsonl")
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_ROOT / "manifest.v1.json")
    parser.add_argument("--downloaded-at-utc", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not UTC_RE.fullmatch(args.downloaded_at_utc):
        raise ValueError("download time must be an explicit UTC timestamp")
    rows = select_rows(load_source(args.source))
    write_subset(args.out, rows)
    manifest = build_manifest(
        DEFAULT_ROOT, args.source, args.out, rows, args.downloaded_at_utc)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "qualified": True,
        "subset": str(args.out),
        "subset_sha256": sha256(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
