"""simulate(): one movement recording in, one synthetic sEMG recording out.

Output file (HDF5):

    emg              (N, channels)   device output (raw: mV; otto_bock: V envelope; myo: int8 counts), attrs: fs
    labels/<name>    (T, ...)        at 100 Hz, attrs: fs. q, qd, q_ref (joint angles, rad), latent, act, exc
                                     (muscle activation / excitation), reserve, cocontraction, grip_load,
                                     restimulus, fibre_len
    spikes/...                       biomime + save_spikes: (unit, sample @ 2048 Hz) pairs and the unit table
    attrs                            config, subject_params, session_params, meta, qc (JSON), myogen_version

Labels cover exactly the input's duration; the EMG has duration * fs_emg samples.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..body.model import shared_model
from ..motion.recording import Motion
from ..subject import sample_session, sample_subject
from .config import SimConfig
from .trial import FS_CTRL, given_motion, simulate_trial, write_recording

log = logging.getLogger(__name__)


@dataclass
class SimResult:
    emg: np.ndarray
    fs_emg: float
    labels: dict
    fs_label: float
    qc: dict
    config: dict
    subject_params: dict
    session_params: dict
    meta: dict
    spikes: dict | None = field(default=None, repr=False)

    def save(self, path):
        """Write the HDF5 output file (format: module docstring)."""
        import h5py

        from .. import __version__
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            write_recording(f, self.emg, self.labels, self.spikes, self.meta.get("spike_muscles"))
            f["emg"].attrs["fs"] = self.fs_emg
            f["labels"].attrs["fs"] = self.fs_label
            f.attrs.update({"config": json.dumps(self.config), "subject_params": json.dumps(self.subject_params),
                            "session_params": json.dumps(self.session_params), "meta": json.dumps(self.meta),
                            "qc": json.dumps(self.qc), "myogen_version": __version__})
        return path


def simulate(motion, config=None, out=None):
    """Simulate the sEMG of `motion` (a Motion or a motion file) with `config` (a SimConfig, a dict in the YAML
    layout, a YAML file, or None for the defaults). Writes `out` (HDF5) when given. Returns a SimResult."""
    motion = motion if isinstance(motion, Motion) else Motion.load(motion)
    cfg = SimConfig.coerce(config)
    cfg.validate()                                     # fields may have changed since construction
    engine, device = cfg.engine, cfg.device
    bank_path = engine.biomime.bank_path() if engine.name == "biomime" else None

    _, _, anat = shared_model()
    # Independent random streams: subject prior, session prior, the trial, config overrides.
    sub_rng, ses_rng, trial_rng, orng = (np.random.default_rng(np.random.SeedSequence([cfg.seed, k]))
                                         for k in (0, 1, 2, 3))
    subject = sample_subject(sub_rng, len(anat.muscle_names), anat.extrinsic, device.electrode_spec(),
                             amputee=cfg.subject.amputee, device=device.name, overrides=cfg.subject.overrides(),
                             orng=orng)
    session = sample_session(ses_rng, subject, support=cfg.session.support, overrides=cfg.session.overrides(),
                             orng=orng)
    synth = None
    if engine.name == "biomime":
        from ..emg.biomime.bank import MuapBank
        from ..emg.biomime.synth import BioMimeEMG
        synth = BioMimeEMG(MuapBank(bank_path), subject, session, anat, pool=engine.biomime.pool,
                           gain=engine.biomime.gain)

    log.info(f"simulating {motion.duration:.1f} s ({engine.name} engine, {device.name} device, "
             f"{subject.layout.n_channels} channels)")
    t0 = time.time()
    trial = simulate_trial(subject, session, given_motion(*motion.resampled(FS_CTRL)), trial_rng,
                           fs_out=device.fs_out, biomime=synth, save_spikes=engine.biomime.save_spikes)
    if trial.spikes is not None:
        trial.meta["spike_muscles"] = list(synth.muscles)
    if not trial.qc["pass"]:
        log.warning(f"the simulated hand did not follow the motion well (tracking RMSE {trial.qc['rmse_deg']:.1f} "
                    f"deg, saturation {trial.qc['saturation']:.0%}): see result.qc")
    log.info(f"done in {time.time() - t0:.0f} s")
    result = SimResult(emg=trial.emg, fs_emg=device.fs_out, labels=trial.labels, fs_label=FS_CTRL, qc=trial.qc,
                       config=cfg.to_dict(), subject_params=subject.to_dict(), session_params=session.to_dict(),
                       meta=trial.meta, spikes=trial.spikes)
    if out is not None:
        result.save(out)
        log.info(f"wrote {out}")
    return result

