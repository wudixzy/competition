import contextlib
import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)
_ENABLED = os.getenv("BI100_PROFILE", "0") == "1"
_INCLUDE_STARTUP = os.getenv("BI100_PROFILE_INCLUDE_STARTUP", "0") == "1"


@contextlib.contextmanager
def bi100_timer(name: str):
    if (not _ENABLED
            or (not _INCLUDE_STARTUP
                and os.getenv("BI100_IN_STARTUP_PROFILE") == "1")):
        yield
        return
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        logger.info("[BI100_PROFILE] %s %.3f ms", name,
                    (time.perf_counter() - t0) * 1000)
