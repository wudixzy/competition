import contextlib
import fnmatch
import json
import os
import threading
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
_ENABLED = os.getenv("BI100_PROFILE", "0") == "1"
_INCLUDE_STARTUP = os.getenv("BI100_PROFILE_INCLUDE_STARTUP", "0") == "1"
_MODE = os.getenv("BI100_PROFILE_MODE", "sync").strip().lower()
_FILTERS = tuple(
    item.strip()
    for item in os.getenv("BI100_PROFILE_FILTER", "").split(",")
    if item.strip()
)
if _ENABLED and _MODE not in {"sync", "event"}:
    raise RuntimeError(f"unsupported BI100_PROFILE_MODE={_MODE!r}")

_EVENT_RECORDS = []
_COUNTERS = {}
_LOCK = threading.Lock()
_FORWARD_INDEX = 0
_LAST_FLUSH_NS = None


def _enabled_for(name: str) -> bool:
    return (_ENABLED
            and (not _FILTERS
                 or any(fnmatch.fnmatchcase(name, pattern)
                        for pattern in _FILTERS)))


def _skip_startup() -> bool:
    return (not _INCLUDE_STARTUP
            and os.getenv("BI100_IN_STARTUP_PROFILE") == "1")


def bi100_profile_event_enabled() -> bool:
    return _ENABLED and _MODE == "event" and not _skip_startup()


def bi100_profile_count(name: str, **metadata) -> None:
    """Record privacy-safe path metadata for the current model forward."""
    if not bi100_profile_event_enabled() or not _enabled_for(name):
        return
    normalized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise TypeError("profile counter keys must be non-empty strings")
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, str) and len(value) <= 64:
            normalized[key] = value
        else:
            raise TypeError(
                "profile counter values must be bool, int, or short strings")
    encoded = json.dumps(
        {"name": name, **normalized}, sort_keys=True, separators=(",", ":"))
    with _LOCK:
        _COUNTERS[encoded] = _COUNTERS.get(encoded, 0) + 1


@contextlib.contextmanager
def bi100_timer(name: str):
    if not _enabled_for(name) or _skip_startup():
        yield
        return
    import torch

    if _MODE == "event":
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        host_started_ns = time.monotonic_ns()
        started.record()
        try:
            yield
        finally:
            finished.record()
            with _LOCK:
                _EVENT_RECORDS.append(
                    (name, started, finished, host_started_ns))
        return

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        logger.info("[BI100_PROFILE] %s %.3f ms", name,
                    (time.perf_counter() - t0) * 1000)


def bi100_profile_flush(**metadata):
    """Synchronize once and emit one aggregate event record per model forward."""
    global _FORWARD_INDEX, _LAST_FLUSH_NS
    if not bi100_profile_event_enabled():
        return None

    with _LOCK:
        records = list(_EVENT_RECORDS)
        counters = dict(_COUNTERS)
        _EVENT_RECORDS.clear()
        _COUNTERS.clear()
    if not records:
        return None

    import torch

    torch.cuda.synchronize()
    flushed_ns = time.monotonic_ns()
    regions = {}
    model_started_ns = None
    for name, started, finished, host_started_ns in records:
        stats = regions.setdefault(name, {"count": 0, "total_ms": 0.0})
        stats["count"] += 1
        stats["total_ms"] += float(started.elapsed_time(finished))
        if name == "model.forward":
            model_started_ns = (host_started_ns if model_started_ns is None
                                else min(model_started_ns, host_started_ns))

    counter_rows = []
    for encoded, count in sorted(counters.items()):
        row = json.loads(encoded)
        row["count"] = count
        counter_rows.append(row)

    payload = {
        "forward_index": _FORWARD_INDEX,
        "metadata": metadata,
        "event_count": len(records),
        "regions": regions,
        "counters": counter_rows,
        "host_model_to_flush_ms": (
            (flushed_ns - model_started_ns) / 1_000_000
            if model_started_ns is not None else None),
        "host_gap_since_previous_flush_ms": (
            (model_started_ns - _LAST_FLUSH_NS) / 1_000_000
            if model_started_ns is not None and _LAST_FLUSH_NS is not None
            else None),
    }
    _FORWARD_INDEX += 1
    _LAST_FLUSH_NS = flushed_ns
    logger.info("[BI100_PROFILE_EVENT] %s",
                json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload
