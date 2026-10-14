from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .inputs.electrodes import Electrodes
from .inputs.subject import Subject


@dataclass
class Recording:
    """The output of a simulation: the device's EMG plus what the hand and muscles did (labels)."""
    emg: np.ndarray                       # (N, C) device output
    fs: float
    labels: dict[str, np.ndarray]         # (T, ...) per label, at fs_labels
    fs_labels: float
    joint_names: list[str]
    muscle_names: list[str]
    electrodes: Electrodes
    subject: Subject
    qc: dict[str, Any]
    filled_joints: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.emg) / self.fs

    def save(self, path: str | Path) -> Path:
        import h5py

        from . import __version__
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("emg", data=self.emg, compression="gzip", compression_opts=4, chunks=True)
            f["emg"].attrs.update({"fs": self.fs, "channel_names": self.electrodes.names})
            for k, v in self.labels.items():
                f.create_dataset(f"labels/{k}", data=np.asarray(v, np.float32), compression="gzip")
            f["labels"].attrs.update({"fs": self.fs_labels, "joint_names": self.joint_names,
                                      "muscle_names": self.muscle_names})
            f.attrs.update({"electrodes": json.dumps(self.electrodes.to_dict()),
                            "subject": json.dumps(self.subject.to_dict()), "qc": json.dumps(self.qc),
                            "filled_joints": json.dumps(self.filled_joints), "myogen_version": __version__})
        return path

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        import h5py
        with h5py.File(path, "r") as f:
            return cls(emg=f["emg"][()], fs=float(f["emg"].attrs["fs"]),
                       labels={k: f["labels"][k][()] for k in f["labels"]}, fs_labels=float(f["labels"].attrs["fs"]),
                       joint_names=[str(n) for n in f["labels"].attrs["joint_names"]],
                       muscle_names=[str(n) for n in f["labels"].attrs["muscle_names"]],
                       electrodes=Electrodes.from_dict(json.loads(f.attrs["electrodes"])),
                       subject=Subject.from_dict(json.loads(f.attrs["subject"])),
                       qc=json.loads(f.attrs["qc"]), filled_joints=json.loads(f.attrs["filled_joints"]))
