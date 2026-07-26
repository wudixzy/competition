import importlib.util
import logging
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "qwen3_6_scripts" / "block_major_kv_cache.py"


class FakeDevice:

    def __init__(self, device_type):
        self.type = device_type


class FakeTensor:

    def __init__(
        self,
        data,
        *,
        dtype="int64",
        device_type="cpu",
        contiguous=True,
        shape=None,
    ):
        self.data = data
        self.dtype = dtype
        self.device = FakeDevice(device_type)
        self._contiguous = contiguous
        if shape is not None:
            self.shape = shape
        elif data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)

    def is_contiguous(self):
        return self._contiguous

    def dim(self):
        return len(self.shape)

    def tolist(self):
        return self.data

    def contiguous(self):
        return FakeTensor(
            list(self.data),
            dtype=self.dtype,
            device_type=self.device.type,
            contiguous=True,
            shape=self.shape,
        )

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def to(self, *, dtype=None, device=None, non_blocking=False):
        del non_blocking
        if device is None:
            device_type = self.device.type
        elif hasattr(device, "type"):
            device_type = device.type
        else:
            device_type = str(device).split(":", 1)[0]
        return FakeTensor(
            list(self.data),
            dtype=self.dtype if dtype is None else dtype,
            device_type=device_type,
            contiguous=self._contiguous,
            shape=self.shape,
        )

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_selector, column = key
            if not isinstance(row_selector, slice):
                raise TypeError("only row slices are supported")
            rows = self.data[row_selector]
            return FakeTensor(
                [row[column] for row in rows],
                dtype=self.dtype,
                device_type=self.device.type,
            )
        if isinstance(key, slice):
            return FakeTensor(
                self.data[key],
                dtype=self.dtype,
                device_type=self.device.type,
            )
        return self.data[key]


def fake_torch_module():
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = FakeTensor
    torch_module.int64 = "int64"
    torch_module.int32 = "int32"
    torch_module.float16 = "float16"
    return torch_module


def load_module():
    saved = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "vllm",
            "vllm.logger",
            "vllm.block_major_kv_cache",
        )
    }
    sys.modules["torch"] = fake_torch_module()
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = logger
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm.block_major_kv_cache", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


module = load_module()
torch = module.torch


class FakeBuffer:

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, key):
        del key
        return self

    def copy_(self, source, non_blocking=False):
        self.calls.append(
            ("copy", source.name, self.name, non_blocking))
        return self


class FakeEvent:

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def record(self):
        self.calls.append(("record", self.name))

    def synchronize(self):
        self.calls.append(("synchronize", self.name))


class FakeError:

    def __init__(self, calls):
        self.calls = calls

    def zero_(self):
        self.calls.append(("zero_error",))


class FakeExtension:

    def __init__(self, calls):
        self.calls = calls

    def pack(self, cache, ids, staging, error, count):
        del cache, error
        self.calls.append(
            ("pack", ids.tolist(), staging.name, count))

    def scatter(self, staging, ids, cache, error, count):
        del cache, error
        self.calls.append(
            ("gpu_scatter", staging.name, ids.tolist(), count))

    def cpu_gather(self, pool, ids, staging, count):
        del pool
        self.calls.append(
            ("cpu_gather", ids.tolist(), staging.name, count))

    def cpu_scatter(self, staging, pool, ids, count):
        del pool
        self.calls.append(
            ("cpu_scatter", staging.name, ids.tolist(), count))

    def check_error(self, error):
        del error
        self.calls.append(("check_error",))


def fake_cache():
    calls = []
    cache = module.BlockMajorCpuKVCache.__new__(
        module.BlockMajorCpuKVCache)
    cache.extension = FakeExtension(calls)
    cache.gpu_cache = ["gpu_cache"]
    cache.cpu_pool = "cpu_pool"
    cache.cpu_staging = [
        FakeBuffer(f"cpu{index}", calls) for index in range(2)
    ]
    cache.gpu_staging = [
        FakeBuffer(f"gpu{index}", calls) for index in range(2)
    ]
    cache.events = [
        FakeEvent(f"event{index}", calls) for index in range(2)
    ]
    cache.error_flag = FakeError(calls)
    cache.num_gpu_blocks = 4096
    cache.num_cpu_blocks = 4096
    cache.trace_enabled = False
    cache._to_gpu_ids = lambda ids: ids.to(dtype=torch.int32)
    return cache, calls


class BlockMajorKVCacheTest(unittest.TestCase):

    def test_selectors_are_strict_and_default_off(self):
        self.assertFalse(module.block_major_cpu_kv_enabled({}))
        self.assertFalse(module.block_major_cpu_kv_enabled({
            module.ENABLE_ENV: "0",
        }))
        self.assertTrue(module.block_major_cpu_kv_enabled({
            module.ENABLE_ENV: "1",
        }))
        self.assertTrue(module.block_major_cpu_kv_trace_enabled({
            module.TRACE_ENV: "1",
        }))
        for invalid in ("", "true", "on", "2", " 1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                        RuntimeError, "exactly '0' or '1'"):
                    module.block_major_cpu_kv_enabled({
                        module.ENABLE_ENV: invalid,
                    })

    def test_gpu_capacity_reserve_is_exact_and_default_off(self):
        profiled_blocks = 67_512
        self.assertEqual(
            module.reserve_block_major_gpu_blocks(
                profiled_blocks,
                module.BYTES_PER_BLOCK,
                {},
            ),
            profiled_blocks,
        )
        candidate = {
            module.ENABLE_ENV: "1",
            module.CPU_OFFLOAD_ENV: "1",
            module.HYBRID_ACCOUNTING_ENV: "full_attention",
        }
        self.assertEqual(module.GPU_STAGING_BYTES, 167_772_160)
        self.assertEqual(
            module.reserve_block_major_gpu_blocks(
                profiled_blocks,
                module.BYTES_PER_BLOCK,
                candidate,
            ),
            profiled_blocks - 1024,
        )

    def test_gpu_capacity_reserve_fails_closed(self):
        valid = {
            module.ENABLE_ENV: "1",
            module.CPU_OFFLOAD_ENV: "1",
            module.HYBRID_ACCOUNTING_ENV: "full_attention",
        }
        with self.assertRaisesRegex(RuntimeError, "CPU_KV_OFFLOAD=1"):
            module.reserve_block_major_gpu_blocks(
                2048,
                module.BYTES_PER_BLOCK,
                {**valid, module.CPU_OFFLOAD_ENV: "0"},
            )
        with self.assertRaisesRegex(RuntimeError, "full_attention"):
            module.reserve_block_major_gpu_blocks(
                2048,
                module.BYTES_PER_BLOCK,
                {**valid, module.HYBRID_ACCOUNTING_ENV: "legacy40"},
            )
        with self.assertRaisesRegex(RuntimeError, "cache block size"):
            module.reserve_block_major_gpu_blocks(
                2048,
                module.BYTES_PER_BLOCK * 4,
                valid,
            )
        with self.assertRaisesRegex(RuntimeError, "no usable GPU KV blocks"):
            module.reserve_block_major_gpu_blocks(
                1024,
                module.BYTES_PER_BLOCK,
                valid,
            )

    def test_mapping_validation_returns_columns(self):
        mapping = FakeTensor([[1, 7], [3, 2], [8, 5]])
        source, destination = module.validate_block_mapping(
            mapping, source_limit=9, destination_limit=8)
        self.assertEqual(source.tolist(), [1, 3, 8])
        self.assertEqual(destination.tolist(), [7, 2, 5])
        self.assertTrue(source.is_contiguous())
        self.assertTrue(destination.is_contiguous())

        empty = FakeTensor([], shape=(0, 2))
        source, destination = module.validate_block_mapping(empty, 1, 1)
        self.assertEqual(source.numel(), 0)
        self.assertEqual(destination.numel(), 0)

    def test_mapping_validation_fails_before_transfer(self):
        invalid_cases = [
            (FakeTensor([[0, 0], [0, 1]]),
             "duplicate source"),
            (FakeTensor([[0, 0], [1, 0]]),
             "duplicate destination"),
            (FakeTensor([[-1, 0]]),
             "source block out of range"),
            (FakeTensor([[0, 4]]),
             "destination block out of range"),
            (FakeTensor([[0, 0]], dtype="int32"),
             "torch.int64"),
            (FakeTensor([0, 0]),
             "shape"),
            (FakeTensor([[0, 1], [2, 3]], contiguous=False),
             "contiguous"),
        ]
        for mapping, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        (TypeError, ValueError), message):
                    module.validate_block_mapping(mapping, 4, 4)

        cache, calls = fake_cache()
        with self.assertRaisesRegex(ValueError, "source block out of range"):
            cache.swap_out(FakeTensor([[4096, 0]]))
        self.assertEqual(calls, [])

    def test_double_buffer_swap_out_sequence(self):
        cache, calls = fake_cache()
        mapping = FakeTensor([
            [index, 1024 - index] for index in range(1025)
        ])
        cache.swap_out(mapping)

        packs = [call for call in calls if call[0] == "pack"]
        scatters = [
            call for call in calls if call[0] == "cpu_scatter"
        ]
        self.assertEqual([call[3] for call in packs], [512, 512, 1])
        self.assertEqual(
            [call[3] for call in scatters], [512, 512, 1])
        self.assertEqual([call[2] for call in packs], [
            "gpu0", "gpu1", "gpu0",
        ])
        self.assertEqual(calls[0], ("zero_error",))
        self.assertEqual(calls[-1], ("check_error",))
        self.assertLess(
            calls.index(("synchronize", "event0")),
            next(index for index, call in enumerate(calls)
                 if call[0] == "cpu_scatter"),
        )

    def test_double_buffer_swap_in_guards_slot_reuse(self):
        cache, calls = fake_cache()
        mapping = FakeTensor([
            [1024 - index, index] for index in range(1025)
        ])
        cache.swap_in(mapping)

        gathers = [
            call for call in calls if call[0] == "cpu_gather"
        ]
        scatters = [
            call for call in calls if call[0] == "gpu_scatter"
        ]
        self.assertEqual([call[3] for call in gathers], [512, 512, 1])
        self.assertEqual([call[3] for call in scatters], [512, 512, 1])
        sync0 = calls.index(("synchronize", "event0"))
        third_gather = [
            index for index, call in enumerate(calls)
            if call[0] == "cpu_gather"
        ][2]
        self.assertLess(sync0, third_gather)
        self.assertEqual(calls[0], ("zero_error",))
        self.assertEqual(calls[-1], ("check_error",))

    def test_fixed_geometry_and_submission_default(self):
        self.assertEqual(module.BYTES_PER_BLOCK, 163840)
        self.assertEqual(module.STAGING_BLOCKS, 512)
        self.assertEqual(module.STAGING_BUFFER_COUNT, 2)
        self.assertEqual(module.GPU_STAGING_BYTES, 167772160)
        run_config = (ROOT / "computility-run.yaml").read_text(
            encoding="utf-8")
        self.assertNotIn(module.ENABLE_ENV, run_config)
        self.assertNotIn(module.TRACE_ENV, run_config)


if __name__ == "__main__":
    unittest.main()
