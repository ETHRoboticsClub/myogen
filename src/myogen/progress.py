from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ProgressFn = Callable[[str, int, int], None]      # (stage, done, total)

STAGES = {"control": "muscles", "emg": "EMG"}     # simulation stage -> what the bar shows


class RunProgress:
    """Rich progress bars for Simulator runs: one row per motion (its current stage), plus a total for batches.
    Disabled, every method is a no-op."""

    def __init__(self, names: list[str], enabled: bool = True) -> None:
        self.names, self.enabled = names, enabled
        self.bars = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                             MofNCompleteColumn(), TextColumn("{task.fields[unit]}"), TimeElapsedColumn(),
                             TextColumn("eta"), TimeRemainingColumn(), disable=not enabled)
        self.rows: dict[int, TaskID] = {}
        self.stage: dict[int, str] = {}
        self.done: set[int] = set()
        self.total: TaskID | None = None

    def __enter__(self) -> RunProgress:
        self.bars.start()
        if len(self.names) > 1:
            self.total = self.bars.add_task(f"[bold]{len(self.names)} motions", total=len(self.names), unit="motions")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.bars.stop()

    def update(self, i: int, stage: str, done: int, total: int) -> None:
        """Motion i is at `done` of `total` ticks of `stage`."""
        if not self.enabled or i in self.done:      # workers' updates can arrive after their result
            return
        description = f"{self.names[i]} [cyan]{STAGES.get(stage, stage)}"
        if i not in self.rows:
            self.rows[i] = self.bars.add_task(description, total=total, unit="ticks")
        elif self.stage.get(i) != stage:       # a new stage: restart the bar (and its clock) for it
            self.bars.reset(self.rows[i], total=total, description=description)
        self.stage[i] = stage
        self.bars.update(self.rows[i], completed=done)

    def reporter(self, i: int) -> ProgressFn:
        """The progress callback of motion i."""
        return lambda stage, done, total: self.update(i, stage, done, total)

    def finished(self, i: int) -> None:
        if not self.enabled:
            return
        self.done.add(i)
        if i in self.rows:
            self.bars.update(self.rows[i], description=f"{self.names[i]} [green]done", visible=len(self.names) <= 8)
        if self.total is not None:
            self.bars.advance(self.total)
