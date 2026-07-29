from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import unittest


ROOT = Path(__file__).resolve().parents[1]
SAMPLING_METADATA = ROOT / "vllm/model_executor/sampling_metadata.py"
SAMPLER = ROOT / "vllm/model_executor/layers/sampler.py"
SAMPLING_PARAMS = ROOT / "vllm/sampling_params.py"
PROTOCOL = ROOT / "qwen3_6_scripts/protocol.py"
DOCKERFILE = ROOT / "Dockerfile"
PATCH_OPS = ROOT / "qwen3_6_scripts/patch_ops.sh"
INSTALLER = ROOT / "scripts/install_bi100_bare_host_runtime.sh"
IDENTITY = ROOT / "tests/verify_bare_host_runtime_identity.py"


class _Tensor:

    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, column = index
            return _Tensor(self.values[row][column])
        return _Tensor(self.values[index])

    def tolist(self):
        if isinstance(self.values, list):
            return self.values
        raise TypeError("scalar test tensor has no list representation")


def _functions(path: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise AssertionError(f"missing helper in {path}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class _SequenceData:

    def __init__(self, computed: int, prompt_tokens: list[int] | None = None):
        self._computed = computed
        self.prompt_token_ids = prompt_tokens or []

    def get_num_computed_tokens(self) -> int:
        return self._computed


class _SamplingType(Enum):
    GREEDY = 1
    RANDOM = 2


class _SequenceGroupToSample:

    def __init__(self, **values):
        self.__dict__.update(values)

    @property
    def do_sample(self) -> bool:
        return bool(self.sample_indices)


def _metadata_helpers() -> dict:
    return _functions(
        SAMPLING_METADATA,
        {
            "_get_prompt_logprob_output_indices",
            "_prepare_seq_groups",
        },
        {
            "Dict": Dict,
            "List": List,
            "Optional": Optional,
            "SamplingParams": object,
            "SamplingMetadataCache": object,
            "SamplingType": _SamplingType,
            "SequenceData": object,
            "SequenceGroupMetadata": object,
            "SequenceGroupToSample": _SequenceGroupToSample,
            "Tuple": Tuple,
            "torch": SimpleNamespace(Generator=object),
        },
    )


def _sampler_helpers() -> dict:

    class _Logprob:

        def __init__(self, logprob: float, rank: int):
            self.logprob = logprob
            self.rank = rank

    return _functions(
        SAMPLER,
        {
            "get_logprobs",
            "_get_next_prompt_tokens",
            "_get_prompt_logprob_if_needed",
            "_get_sampled_logprob_if_needed",
        },
        {
            "Dict": dict,
            "List": list,
            "Optional": object,
            "PromptLogprobs": list,
            "SampleLogprobs": list,
            "SampleResultType": object,
            "SamplingMetadata": object,
            "SequenceGroupToSample": object,
            "Tuple": tuple,
            "Logprob": _Logprob,
            "inf": float("inf"),
            "torch": SimpleNamespace(Tensor=_Tensor),
        },
    )


class SparsePromptLogprobsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = _metadata_helpers()
        cls.sampler = _sampler_helpers()

    def output_indices(
        self,
        *,
        computed: int,
        prompt_logprob_len: int,
        positions: list[int] | None,
        enabled: bool = True,
    ) -> list[int]:
        params = SimpleNamespace(
            prompt_logprobs=5 if enabled else None,
            prompt_logprob_positions=positions,
        )
        return self.metadata["_get_prompt_logprob_output_indices"](
            params,
            _SequenceData(
                computed,
                list(range(computed + prompt_logprob_len + 2)),
            ),
            prompt_logprob_len,
        )

    def test_sparse_positions_cross_chunk_boundaries_without_duplicates(
            self) -> None:
        positions = [1, 8191, 8192, 8193, 16384, 32767]
        self.assertEqual(
            self.output_indices(
                computed=0,
                prompt_logprob_len=8192,
                positions=positions,
            ),
            [0, 8190, 8191],
        )
        self.assertEqual(
            self.output_indices(
                computed=8192,
                prompt_logprob_len=8192,
                positions=positions,
            ),
            [0, 8191],
        )
        self.assertEqual(
            self.output_indices(
                computed=24576,
                prompt_logprob_len=8191,
                positions=positions,
            ),
            [8190],
        )

    def test_empty_sparse_chunk_and_disabled_mode_select_nothing(self) -> None:
        self.assertEqual(
            self.output_indices(
                computed=8192,
                prompt_logprob_len=8192,
                positions=[1, 32767],
            ),
            [],
        )
        self.assertEqual(
            self.output_indices(
                computed=0,
                prompt_logprob_len=8192,
                positions=[1],
                enabled=False,
            ),
            [],
        )

    def test_none_positions_preserves_standard_all_position_behavior(
            self) -> None:
        self.assertEqual(
            self.output_indices(
                computed=8192,
                prompt_logprob_len=3,
                positions=None,
            ),
            [0, 1, 2],
        )

    def test_final_unsampled_chunk_stops_before_missing_next_token(
            self) -> None:
        params = SimpleNamespace(
            prompt_logprobs=5,
            prompt_logprob_positions=None,
        )
        helper = self.metadata["_get_prompt_logprob_output_indices"]
        self.assertEqual(
            helper(params, _SequenceData(8, list(range(10))), 2),
            [0],
        )
        params.prompt_logprob_positions = [9, 10]
        self.assertEqual(
            helper(params, _SequenceData(8, list(range(10))), 2),
            [0],
        )

    def test_next_teacher_tokens_follow_sparse_output_offsets(self) -> None:
        group = SimpleNamespace(
            is_prompt=True,
            seq_ids=[7],
            query_len=8192,
            seq_data={7: _SequenceData(
                8192,
                list(range(20000)),
            )},
            prompt_logprob_output_indices=[0, 8191],
        )
        self.assertEqual(
            self.sampler["_get_next_prompt_tokens"](group),
            [8193, 16384],
        )

    def test_zero_sample_chunk_returns_full_none_placeholders(self) -> None:
        group = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt_logprobs=5),
            is_prompt=True,
            query_len=8192,
            seq_ids=[1],
            do_sample=False,
            seq_data={1: _SequenceData(8192, list(range(20000)))},
            prompt_logprob_indices=[],
            prompt_logprob_output_indices=[],
        )
        result, top_index, selected_index = self.sampler[
            "_get_prompt_logprob_if_needed"
        ](
            group,
            None,
            None,
            None,
            None,
            4,
            6,
        )
        self.assertEqual(len(result), 8192)
        self.assertTrue(all(row is None for row in result))
        self.assertEqual(top_index, 6)
        self.assertEqual(selected_index, 4)

    def test_final_unsampled_chunk_response_has_prompt_length_minus_one(
            self) -> None:
        group = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt_logprobs=5),
            is_prompt=True,
            query_len=2,
            seq_ids=[1],
            do_sample=False,
            seq_data={1: _SequenceData(8, list(range(10)))},
            prompt_logprob_indices=[0],
            prompt_logprob_output_indices=[0],
        )
        result, _, _ = self.sampler["_get_prompt_logprob_if_needed"](
            group,
            _Tensor([-0.2]),
            _Tensor([1]),
            _Tensor([[9, 99, 98, 97, 96]]),
            _Tensor([[-0.2, -1.0, -2.0, -3.0, -4.0]]),
            0,
            0,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {9, 96, 97, 98, 99})

    def test_empty_query_does_not_construct_invalid_gpu_indices(self) -> None:
        group = SimpleNamespace(
            sampling_params=SimpleNamespace(
                prompt_logprobs=5,
                logprobs=None,
            ),
            is_prompt=True,
            query_len=8192,
            seq_ids=[1],
            do_sample=False,
            seq_data={1: _SequenceData(8192, list(range(20000)))},
            prompt_logprob_indices=[],
            prompt_logprob_output_indices=[],
        )
        metadata = SimpleNamespace(seq_groups=[group])
        prompt, sampled = self.sampler["get_logprobs"](
            object(),
            metadata,
            [([], [])],
        )
        self.assertEqual(len(prompt), 1)
        self.assertEqual(len(prompt[0]), 8192)
        self.assertTrue(all(row is None for row in prompt[0]))
        self.assertEqual(sampled, [[]])

    def test_selected_rows_retain_full_chunk_shape(self) -> None:
        group = SimpleNamespace(
            sampling_params=SimpleNamespace(prompt_logprobs=2),
            is_prompt=True,
            query_len=4,
            seq_ids=[1],
            do_sample=False,
            seq_data={1: _SequenceData(0, [10, 11, 12, 13, 14])},
            prompt_logprob_indices=[0, 1],
            prompt_logprob_output_indices=[0, 3],
        )
        result, top_index, selected_index = self.sampler[
            "_get_prompt_logprob_if_needed"
        ](
            group,
            _Tensor([-0.2, -0.3]),
            _Tensor([1, 2]),
            _Tensor([[11, 99], [14, 98]]),
            _Tensor([[-0.2, -1.0], [-0.1, -0.3]]),
            0,
            0,
        )
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertEqual(set(result[0]), {11, 99})
        self.assertEqual(set(result[3]), {14, 98})
        self.assertEqual(top_index, 2)
        self.assertEqual(selected_index, 2)

    def test_mixed_batch_accumulates_sparse_full_and_sample_indices(
            self) -> None:
        sparse = SimpleNamespace(
            prompt_logprobs=5,
            prompt_logprob_positions=[1, 3],
            seed=None,
            sampling_type=_SamplingType.GREEDY,
        )
        full = SimpleNamespace(
            prompt_logprobs=5,
            prompt_logprob_positions=None,
            seed=None,
            sampling_type=_SamplingType.GREEDY,
        )
        groups = [
            SimpleNamespace(
                seq_data={1: _SequenceData(0, list(range(10)))},
                sampling_params=sparse,
                is_prompt=True,
                do_sample=True,
                request_id="sparse",
            ),
            SimpleNamespace(
                seq_data={2: _SequenceData(0, list(range(10)))},
                sampling_params=full,
                is_prompt=True,
                do_sample=False,
                request_id="full",
            ),
        ]
        prepared, selected, categorized, prompts = self.metadata[
            "_prepare_seq_groups"
        ](
            groups,
            [4, 3],
            [4, 3],
            "cpu",
        )
        self.assertEqual(prompts, 2)
        self.assertEqual(selected, [0, 2, 3, 4, 5, 6])
        self.assertEqual(
            categorized[_SamplingType.GREEDY],
            [2],
        )
        self.assertEqual(prepared[0].prompt_logprob_indices, [0, 1])
        self.assertEqual(prepared[0].prompt_logprob_output_indices, [0, 2])
        self.assertEqual(prepared[0].sample_indices, [2])
        self.assertEqual(prepared[1].prompt_logprob_indices, [3, 4, 5])
        self.assertEqual(
            prepared[1].prompt_logprob_output_indices,
            [0, 1, 2],
        )
        self.assertEqual(prepared[1].sample_indices, [])

    def test_runtime_overlay_contains_all_sparse_engine_files(self) -> None:
        sampling_params = SAMPLING_PARAMS.read_text(encoding="utf-8")
        protocol = PROTOCOL.read_text(encoding="utf-8")
        self.assertIn(
            "prompt_logprob_positions: Optional[List[int]] = None",
            sampling_params,
        )
        self.assertIn(
            "prompt_logprob_positions=(",
            protocol,
        )
        for path in (DOCKERFILE, PATCH_OPS, INSTALLER, IDENTITY):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("sampling_params.py", source)
                self.assertIn("sampling_metadata.py", source)
                self.assertIn("model_executor/layers/sampler.py", source)


if __name__ == "__main__":
    unittest.main()
