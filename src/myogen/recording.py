"""Recording: the output of a simulation, and its HDF5 file.

    emg                (N, C)   device output (raw: mV; myo: int8 counts; otto_bock: V), attrs: fs, channel_names
    labels/<name>      (T, ...) at 100 Hz, attrs: fs, joint_names, muscle_names
                       q, q_ref, qd (joint angles / velocities, rad), excitation, activation, fibre_length,
                       reserve (Nm), effort (co-activation added to the EMG drive), grip_load
    spikes/...                  BioMime with save_spikes: (unit, sample @ 2048 Hz) and the unit table
    attrs                       engine, electrodes, subject, qc (JSON), filled_joints, myogen_version
The labels cover exactly the motion's duration; the EMG has duration * fs samples.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .electrodes import Electrodes
from .subject import Subject


@dataclass
class Recording:
    emg: np.ndarray
    fs: float
    labels: dict
    fs_labels: float
    joint_names: list
    muscle_names: list
    electrodes: Electrodes
    subject: Subject
    engine: str
    qc: dict
    filled_joints: list = field(default_factory=list)
    spikes: dict | None = field(default=None, repr=False)

    @property
    def duration(self):
        return len(self.emg) / self.fs

    def save(self, path):
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
            for k, v in (self.spikes or {}).items():
                f.create_dataset(f"spikes/{k}", data=v, compression="gzip")
            f.attrs.update({"engine": self.engine, "electrodes": json.dumps(self.electrodes.to_dict()),
                            "subject": json.dumps(self.subject.to_dict()), "qc": json.dumps(self.qc),
                            "filled_joints": json.dumps(self.filled_joints), "myogen_version": __version__})
        return path

    @classmethod
    def load(cls, path):
        import h5py
        with h5py.File(path, "r") as f:
            spikes = None
            if "spikes" in f:
                spikes = {}
                f["spikes"].visititems(lambda k, v: spikes.__setitem__(k, v[()]) if hasattr(v, "shape") else None)
            return cls(emg=f["emg"][()], fs=float(f["emg"].attrs["fs"]),
                       labels={k: f["labels"][k][()] for k in f["labels"]}, fs_labels=float(f["labels"].attrs["fs"]),
                       joint_names=[str(n) for n in f["labels"].attrs["joint_names"]],
                       muscle_names=[str(n) for n in f["labels"].attrs["muscle_names"]],
                       electrodes=Electrodes.from_dict(json.loads(f.attrs["electrodes"])),
                       subject=Subject.from_dict(json.loads(f.attrs["subject"])), engine=str(f.attrs["engine"]),
                       qc=json.loads(f.attrs["qc"]), filled_joints=json.loads(f.attrs["filled_joints"]),
                       spikes=spikes)
