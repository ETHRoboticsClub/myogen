from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.context import BaseContext
from typing import Any


def n_workers(n_tasks: int, workers: int | None) -> int:
    """How many processes for n_tasks: `workers` (default: the CPU count), at most one per task."""
    return max(1, min(n_tasks, workers or os.cpu_count() or 1))


def mp_context() -> BaseContext:
    """Worker processes forked from a server process that has myogen imported: they start in milliseconds, where
    spawned ones import myogen (~1 s) one after the other. Forking this process instead is unsafe (MuJoCo, BLAS
    and Jupyter threads); where there is no fork server (Windows), workers are spawned."""
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("spawn")
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload(["myogen"])
    return ctx


def process_pool(workers: int, initializer: Callable[..., None], initargs: tuple[Any, ...]) -> ProcessPoolExecutor:
    """A pool of `workers` processes (see mp_context), each set up by initializer(*initargs)."""
    return ProcessPoolExecutor(workers, mp_context=mp_context(), initializer=initializer, initargs=initargs)
