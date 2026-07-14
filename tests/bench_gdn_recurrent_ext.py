#!/usr/bin/env python3
import argparse
import hashlib
import json
import statistics
import time

import torch

import bi100_gdn_recurrent


HEADS = 12
DIM = 128


def baseline(query, key, value, beta, decay, state):
    state.mul_(decay[:, :, None, None])
    flat = state.view(-1, DIM, DIM)
    head_count = flat.shape[0]
    key_state = torch.bmm(key.view(head_count, 1, DIM), flat).view(
        1, HEADS, DIM
    )
    delta = (value - key_state) * beta[:, :, None]
    flat.baddbmm_(key.view(head_count, DIM, 1),
                  delta.view(head_count, 1, DIM))
    return torch.bmm(query.view(head_count, 1, DIM), flat).view(
        1, HEADS, DIM
    )


def candidate(query, key, value, beta, decay, state):
    return bi100_gdn_recurrent.recurrent_update(
        query, key, value, beta, decay, state
    )


def digest(tensor):
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def error_report(reference, actual):
    difference = (reference - actual).abs()
    return {
        "equal": torch.equal(reference, actual),
        "allclose": torch.allclose(reference, actual, rtol=1e-5, atol=1e-6),
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "reference_hash": digest(reference),
        "candidate_hash": digest(actual),
    }


def make_inputs(steps, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(20260714)
    shape = (steps, 1, HEADS, DIM)
    query = torch.randn(shape, generator=generator, device=device)
    key = torch.randn(shape, generator=generator, device=device)
    query = torch.nn.functional.normalize(query, dim=-1) * (DIM**-0.5)
    key = torch.nn.functional.normalize(key, dim=-1)
    value = torch.randn(shape, generator=generator, device=device) * 0.05
    beta = torch.sigmoid(
        torch.randn(steps, 1, HEADS, generator=generator, device=device)
    )
    decay = 0.97 + 0.025 * torch.rand(
        steps, 1, HEADS, generator=generator, device=device
    )
    state = torch.randn(
        1, HEADS, DIM, DIM, generator=generator, device=device
    ) * 0.02
    return tuple(item.contiguous()
                 for item in (query, key, value, beta, decay, state))


def run_sequence(function, inputs, state):
    output = None
    for step in range(inputs[0].shape[0]):
        output = function(*(item[step] for item in inputs), state)
    return output


def timed_run(function, inputs, initial_state):
    state = initial_state.clone()
    run_sequence(function, tuple(item[:50] for item in inputs), state)
    torch.cuda.synchronize()
    state.copy_(initial_state)
    started = time.perf_counter()
    run_sequence(function, inputs, state)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main():
    args = parse_args()
    query, key, value, beta, decay, initial_state = make_inputs(
        args.steps, args.device
    )
    inputs = (query, key, value, beta, decay)
    input_snapshots = tuple(item.clone() for item in inputs)

    reference_state = initial_state.clone()
    candidate_state = initial_state.clone()
    reference_output = run_sequence(baseline, inputs, reference_state)
    candidate_output = run_sequence(candidate, inputs, candidate_state)
    torch.cuda.synchronize()

    baseline_times = []
    candidate_times = []
    for repeat in range(args.repeats):
        order = (("baseline", baseline), ("candidate", candidate))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, function in order:
            elapsed = timed_run(function, inputs, initial_state)
            target = baseline_times if name == "baseline" else candidate_times
            target.append(elapsed)

    baseline_median = statistics.median(baseline_times)
    candidate_median = statistics.median(candidate_times)
    report = {
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "shape": [1, HEADS, DIM, DIM],
        "steps": args.steps,
        "state": error_report(reference_state, candidate_state),
        "output": error_report(reference_output, candidate_output),
        "baseline_ms": baseline_times,
        "candidate_ms": candidate_times,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "speedup": baseline_median / candidate_median,
        "inputs_unchanged": all(
            torch.equal(before, after)
            for before, after in zip(input_snapshots, inputs)
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")

    passed = (report["state"]["allclose"]
              and report["output"]["allclose"]
              and report["inputs_unchanged"]
              and report["speedup"] >= 1.5)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
