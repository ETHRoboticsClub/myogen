from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import TracebackType

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

ProgressFn = Callable[[str, int, int], None]      # (stage, done, total)

STAGES = {"control": "muscles", "emg": "EMG"}     # simulation stage -> what the row shows
# Share of a motion's simulation time per stage (profiled: the controller ~65%, the EMG ~35%), so one bar spans both.
SPAN = {"control": (0.0, 0.65), "emg": (0.65, 1.0)}

_shown: Progress | None = None                    # the display on screen, which nested calls add their rows to


class _Status(SpinnerColumn):
    """A spinner while running, a dot while queued, a tick when done."""

    def render(self, task: Task) -> Text:
        if not task.started:
            return Text("·", style="dim")
        return Text("✓", style="green") if task.finished else super().render(task)  # type: ignore[return-value]


@contextmanager
def progress_bars(enabled: bool = True) -> Iterator[Progress]:
    """The rich progress display: a new one, or the one already on screen (rich shows one at a time), so a function
    that shows progress can call others that do. Every row has the fields `stage` and `detail`."""
    global _shown
    if enabled and _shown is not None:
        yield _shown
        return
    bars = Progress(_Status(), TextColumn("{task.description}"), TextColumn("{task.fields[stage]}"), BarColumn(),
                    TaskProgressColumn(), TextColumn("{task.fields[detail]}"), TimeElapsedColumn(),
                    TextColumn("eta"), TimeRemainingColumn(), disable=not enabled)
    if not enabled:
        yield bars
        return
    _shown = bars
    try:
        with bars:
            yield bars
    finally:
        _shown = None


class Counter:
    """One row counting `total` steps of something."""

    def __init__(self, bars: Progress, description: str, total: int, unit: str) -> None:
        self.bars, self.total, self.unit, self.n = bars, total, unit, 0
        self.task = bars.add_task(description, total=total, stage="", detail=f"0/{total} {unit}")

    def advance(self) -> None:
        self.n += 1
        self.bars.update(self.task, advance=1, detail=f"{self.n}/{self.total} {self.unit}")

    def close(self) -> None:
        self.bars.update(self.task, completed=self.total, stage="[green]done")


class RunProgress:
    """The rows of a Simulator run: one per motion (queued, then its stage and ticks, then done) and, for a batch, a
    total. Batches of more than max_rows motions show only the running ones."""

    def __init__(self, names: list[str], enabled: bool = True, max_rows: int = 12) -> None:
        self.names, self.enabled, self.max_rows = names, enabled, max_rows
        self.all_rows = len(names) <= max_rows
        self.rows: dict[int, TaskID] = {}
        self.started: set[int] = set()
        self.done: set[int] = set()
        self.total: Counter | None = None

    def __enter__(self) -> RunProgress:
        self.nested = self.enabled and _shown is not None
        self._display = progress_bars(self.enabled)
        self.bars = self._display.__enter__()
        if len(self.names) > 1:
            self.total = Counter(self.bars, f"[bold]{len(self.names)} motions", len(self.names), "done")
        if self.all_rows:
            for i in range(len(self.names)):
                self._add(i)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        if self.nested:            # inside another function's display: leave only the total
            for task in self.rows.values():
                self.bars.remove_task(task)
        self._display.__exit__(exc_type, exc, tb)

    def _add(self, i: int) -> None:
        self.rows[i] = self.bars.add_task(f"  {self.names[i]}" if self.total else self.names[i], total=100,
                                          start=False, stage="[dim]queued", detail="")

    def update(self, i: int, stage: str, done: int, total: int) -> None:
        """Motion i is at `done` of `total` ticks of `stage`."""
        if not self.enabled or i in self.done:      # workers' updates can arrive after their result
            return
        if i not in self.rows:
            self._add(i)
        task = self.rows[i]
        if i not in self.started:
            self.started.add(i)
            self.bars.start_task(task)               # its clock starts when its worker does
        lo, hi = SPAN.get(stage, (0.0, 1.0))
        self.bars.update(task, completed=100 * (lo + (hi - lo) * done / max(total, 1)),
                         stage=f"[cyan]{STAGES.get(stage, stage)}", detail=f"{done}/{total} ticks")

    def reporter(self, i: int) -> ProgressFn:
        """The progress callback of motion i."""
        return lambda stage, done, total: self.update(i, stage, done, total)

    def finished(self, i: int) -> None:
        if not self.enabled:
            return
        self.done.add(i)
        if i in self.rows:
            self.bars.update(self.rows[i], completed=100, stage="[green]done", visible=self.all_rows)
        if self.total is not None:
            self.total.advance()
