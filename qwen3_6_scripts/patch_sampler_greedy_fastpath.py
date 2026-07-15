from __future__ import annotations

import ast

from patch_utils import ensure_file, package_root


SAMPLER = package_root("vllm") / "model_executor" / "layers" / "sampler.py"

IMPORT_ANCHOR = """\
import vllm.envs as envs
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
"""

PATCHED_IMPORT = """\
import vllm.envs as envs
from vllm.bi100_env import env_bool
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
"""

ORIGINAL_BLOCK = """\
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
"""

PATCHED_BLOCK = """\
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Greedy argmax is invariant under positive temperature scaling,
        # top-k/top-p/min-p filtering, FP16-to-FP32 conversion, softmax, and
        # log-softmax. Min-token and repetition/frequency/presence penalties
        # have already been applied above. Requests needing logprobs, deferred
        # output, GPU probabilities, or non-greedy sampling retain the complete
        # reference path below.
        use_bi100_greedy_fast_path = (
            env_bool("BI100_SAMPLER_GREEDY_FASTPATH", True)
            and not self.include_gpu_probs_tensor
            and not self._should_modify_greedy_probs_inplace
            and not sampling_metadata.skip_sampler_cpu_output
            and all(
                seq_group.sampling_params.sampling_type == SamplingType.GREEDY
                and seq_group.sampling_params.logprobs is None
                and seq_group.sampling_params.prompt_logprobs is None
                for seq_group in sampling_metadata.seq_groups))
        if use_bi100_greedy_fast_path:
            sample_results, _ = _sample(
                logits,
                logits,
                sampling_metadata,
                sampling_tensors,
                include_gpu_probs_tensor=False,
                modify_greedy_probs=False,
            )
            assert not isinstance(sample_results, SampleResultArgsType)
            prompt_logprobs, sample_logprobs = get_logprobs(
                logits, sampling_metadata, sample_results)
            return _build_sampler_output(
                sample_results,
                sampling_metadata,
                prompt_logprobs,
                sample_logprobs,
                on_device_tensors=None,
                skip_sampler_cpu_output=False)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
"""

MARKER = "use_bi100_greedy_fast_path"


def main() -> None:
    sampler = ensure_file(SAMPLER)
    source = sampler.read_text()
    if MARKER in source:
        if "from vllm.bi100_env import env_bool" not in source:
            raise RuntimeError(f"incomplete greedy sampler patch: {sampler}")
        print(f"[skip] already patched: {sampler}")
        return
    if IMPORT_ANCHOR not in source:
        raise RuntimeError(f"import anchor not found in {sampler}")
    if ORIGINAL_BLOCK not in source:
        raise RuntimeError(f"sampler block anchor not found in {sampler}")
    patched = source.replace(IMPORT_ANCHOR, PATCHED_IMPORT, 1)
    patched = patched.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1)
    ast.parse(patched, filename=str(sampler))
    sampler.write_text(patched)
    print(f"[ok] patched: {sampler}")


if __name__ == "__main__":
    main()
