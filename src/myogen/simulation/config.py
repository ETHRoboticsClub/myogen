"""SimConfig: everything that describes one simulation besides the motion (template.yml is the commented default).

    SimConfig.load("sim.yml")
    SimConfig(engine="biomime", device="myo", seed=3)       # then set fields: cfg.session.support = "free"
    SimConfig.from_dict({"engine": {"name": "biomime"}})

Loading is strict: unknown keys and wrong types are errors naming the key. Relative paths are relative to the
YAML file. Ranged values: a number fixes the parameter, [lo, hi] draws it with `seed`, null keeps the prior.
"""
import copy
import dataclasses
import math
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import packaged_defaults, user_data_dir
from ..emg.acquisition import DEVICE_FS, DEVICES

TEMPLATE = (Path(__file__).parent / "template.yml").read_text()
ENGINES = ("analytic", "biomime")
POOLS = ("fuglevand", "lif")
SUPPORTS = ("random", "supported", "free")


def electrode_presets():
    """Electrode layout per device: the `electrodes` section of the packaged defaults."""
    return packaged_defaults()["electrodes"]


@dataclass
class BiomimeOptions:
    muap_bank: str | None = None
    pool: str = "fuglevand"
    gain: float = 1.0
    save_spikes: bool = False

    def bank_path(self):
        path = Path(self.muap_bank or user_data_dir() / "biomime" / "muap_bank.h5").expanduser()
        if not path.exists():
            raise FileNotFoundError(f"engine biomime needs the MUAP bank, not found at {path}: set "
                                    f"engine.biomime.muap_bank, or `myogen download muap-bank`")
        return path


@dataclass
class Engine:
    name: str = "analytic"
    biomime: BiomimeOptions = field(default_factory=BiomimeOptions)


@dataclass
class Device:
    name: str = "raw"
    fs: float = 2000
    electrodes: dict = field(default_factory=dict)

    @property
    def fs_out(self):
        return DEVICE_FS.get(self.name, self.fs)

    def electrode_spec(self):
        """The device's preset with `electrodes` applied."""
        return {**copy.deepcopy(electrode_presets()[self.name]), **copy.deepcopy(self.electrodes)}


@dataclass
class SubjectParams:
    amputee: bool = False
    residual_length: float | list | None = None
    forearm_radius_mm: float | list | None = None
    fat_mm: float | list | None = None
    gain_mv: float | list | None = None
    noise_uv: float | list | None = None

    def overrides(self):
        """{Subject field (SI units): value} for subject.sample_subject."""
        return {"residual_length": self.residual_length, "forearm_radius": _scale(self.forearm_radius_mm, 1e-3),
                "fat": _scale(self.fat_mm, 1e-3), "gain_mv": self.gain_mv, "noise_uv": self.noise_uv}


@dataclass
class SessionParams:
    support: str = "random"
    elbow_angle_deg: float | list | None = None
    load: float | list | None = None
    cocontraction: float | list | None = None

    def overrides(self):
        """{Session field (SI units): value} for subject.sample_session."""
        return {"elbow_angle": _scale(self.elbow_angle_deg, math.pi / 180), "load_prob": self.load,
                "cocontraction_mean": self.cocontraction}


@dataclass
class SimConfig:
    seed: int = 0
    engine: Engine = field(default_factory=Engine)
    device: Device = field(default_factory=Device)
    subject: SubjectParams = field(default_factory=SubjectParams)
    session: SessionParams = field(default_factory=SessionParams)

    def __post_init__(self):
        if isinstance(self.engine, str):                # shorthand: SimConfig(engine="biomime", device="myo")
            self.engine = Engine(name=self.engine)
        if isinstance(self.device, str):
            self.device = Device(name=self.device)
        self.validate()

    def validate(self):
        """Raise ValueError naming the offending key."""
        e, d, s, x = self.engine, self.device, self.subject, self.session
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        _choice("engine.name", e.name, ENGINES)
        _choice("engine.biomime.pool", e.biomime.pool, POOLS)
        _number("engine.biomime.gain", e.biomime.gain, lo=0, strict=True)
        _choice("device.name", d.name, DEVICES)
        _number("device.fs", d.fs, lo=0, strict=True)
        if not isinstance(d.electrodes, dict):
            raise ValueError(f"device.electrodes must be a mapping, got {d.electrodes!r}")
        known = set().union(*electrode_presets().values())
        if set(d.electrodes) - known:
            raise ValueError(f"device.electrodes: unknown keys {sorted(set(d.electrodes) - known)}; "
                             f"one of {sorted(known)}")
        if not isinstance(s.amputee, bool):
            raise ValueError(f"subject.amputee must be true or false, got {s.amputee!r}")
        if s.residual_length is not None and not s.amputee:
            raise ValueError("subject.residual_length applies to amputees only (set subject.amputee: true)")
        _ranged("subject.residual_length", s.residual_length, 0.2, 1.0)
        _ranged("subject.forearm_radius_mm", s.forearm_radius_mm, 20, 70)
        _ranged("subject.fat_mm", s.fat_mm, 0, 30)
        _ranged("subject.gain_mv", s.gain_mv, 0, 20, strict=True)
        _ranged("subject.noise_uv", s.noise_uv, 0, 100)
        _choice("session.support", x.support, SUPPORTS)
        _ranged("session.elbow_angle_deg", x.elbow_angle_deg, 0, 150)
        _ranged("session.load", x.load, 0, 1)
        _ranged("session.cocontraction", x.cocontraction, 0, 1)

    @classmethod
    def coerce(cls, config):
        """A SimConfig from a SimConfig, a dict in the YAML layout, a YAML path, or None (the defaults)."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        return cls.load(config)

    @classmethod
    def from_dict(cls, data, base_dir=None):
        """Strict: unknown keys and wrong section types raise ValueError. base_dir: for relative paths."""
        cfg = _build(cls, data or {}, "")
        cfg.validate()
        bank = cfg.engine.biomime.muap_bank
        if bank and base_dir is not None and not Path(bank).expanduser().is_absolute():
            cfg.engine.biomime.muap_bank = str((Path(base_dir) / bank).resolve())
        return cfg

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file {path} not found (SimConfig.write_default writes one)")
        data = yaml.safe_load(path.read_text())
        if data is not None and not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        try:
            return cls.from_dict(data, base_dir=path.parent)
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from None

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return Path(path)

    @staticmethod
    def write_default(path):
        """Write the commented default config (template.yml)."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists")
        path.write_text(TEMPLATE)
        return path


def _build(cls, data, prefix):
    """Instance of dataclass `cls` from a dict, recursing into dataclass-typed fields, without validation."""
    if not isinstance(data, dict):
        raise ValueError(f"{prefix.rstrip('.') or 'config'} must be a mapping, got {data!r}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"unknown key{'s' if len(unknown) > 1 else ''} {', '.join(prefix + k for k in unknown)}; "
                         f"{prefix.rstrip('.') or 'the top level'} has {', '.join(known)}")
    obj = cls.__new__(cls)                    # bypass __post_init__: validated once, by the caller
    for name, f in known.items():
        if name in data:
            value = data[name]
            if dataclasses.is_dataclass(f.type):
                value = _build(f.type, value if value is not None else {}, f"{prefix}{name}.")
        else:
            value = f.default_factory() if f.default_factory is not MISSING else f.default
        setattr(obj, name, value)
    return obj


def _scale(v, k):
    if v is None:
        return None
    return [x * k for x in v] if isinstance(v, (list, tuple)) else v * k


def _choice(key, v, options):
    if v not in options:
        raise ValueError(f"{key} must be one of {' | '.join(options)}, got {v!r}")


def _number(key, v, lo=None, hi=None, strict=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number, got {v!r}")
    if lo is not None and (v < lo or (strict and v == lo)) or hi is not None and v > hi:
        raise ValueError(f"{key} = {v} is outside {'(' if strict else '['}{lo}, {hi}]")


def _ranged(key, v, lo, hi, strict=False):
    """None (the prior), a number, or [lo, hi] with lo <= hi, all within the parameter's [lo, hi]."""
    if v is None:
        return
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError(f"{key} must be a number or [lo, hi], got {v!r}")
        for x in v:
            _number(key, x, lo, hi, strict)
        if v[0] > v[1]:
            raise ValueError(f"{key}: range [lo, hi] needs lo <= hi, got {list(v)}")
        return
    _number(key, v, lo, hi, strict)
