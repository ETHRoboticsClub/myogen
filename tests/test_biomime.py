"""The BioMime engine. Synthesis tests need the MUAP bank (myogen biomime-setup) and PyTorch."""
from pathlib import Path

import numpy as np
import pytest

import myogen as mg
from myogen.body import ForearmModel
from myogen.muscles import MuscleActivity

pytest.importorskip("torch")
from myogen.engines.analytic import AnalyticEngine                    # noqa: E402
from myogen.engines.biomime.anatomy import GridGeometry               # noqa: E402
from myogen.engines.biomime.bank import DEFAULT_BANK                  # noqa: E402
from myogen.engines.biomime.engine import BioMimeEngine, _bilinear, _skin_filter   # noqa: E402
from myogen.engines.biomime.pools import FuglevandPool, LIFPool       # noqa: E402

REPO_BANK = Path(__file__).resolve().parents[1] / "data" / "biomime" / "muap_bank.h5"
BANK = REPO_BANK if REPO_BANK.exists() else DEFAULT_BANK


def test_bilinear_on_a_node_and_wrapping():
    w = _bilinear(5 / 32, 3.0)
    assert np.isclose(w[3 * 32 + 5], 1.0) and np.isclose(w.sum(), 1.0)
    w = _bilinear(31.5 / 32, 0.0)                                    # between the last and the first column
    assert np.isclose(w[31], 0.5) and np.isclose(w[0], 0.5)


def test_skin_filter():
    g = GridGeometry()
    assert np.allclose(_skin_filter(g, 0.003), np.eye(320))
    B = _skin_filter(g, 0.008)
    assert B.sum(1).max() < 1.0 and (B >= 0).all()


@pytest.mark.parametrize("Pool", [FuglevandPool, LIFPool])
def test_pool_recruitment_and_rates(Pool):
    fs, rng = 2048, np.random.default_rng(0)
    pool = Pool(np.linspace(0, 1, 40))
    assert np.all(np.diff(pool.rt) > 0)
    assert sum(len(s) for s in pool.spikes(np.zeros(fs), fs, rng)) == 0
    low = sum(len(s) for s in pool.spikes(np.full(2 * fs, 0.2), fs, rng))
    high = sum(len(s) for s in pool.spikes(np.full(2 * fs, 0.8), fs, rng))
    assert high > 1.5 * low > 0


needs_bank = pytest.mark.skipif(not BANK.exists(), reason="MUAP bank missing (myogen biomime-setup)")


@pytest.fixture(scope="module")
def setup():
    model = ForearmModel()
    ring = mg.Electrodes.ring(n=16, position_cm=7.0)
    engine = BioMimeEngine(BANK, units_per_muscle=24, keep_templates=True)
    engine.prepare(model, mg.Subject(), ring)
    return model, ring, engine


def _activity(model, drive_level, muscle=None, T=150, L=None):
    drive = np.full((T, model.nu), drive_level) if muscle is None else np.zeros((T, model.nu))
    if muscle is not None:
        drive[:, model.muscle_names.index(muscle)] = drive_level
    q = np.tile(model.default_qpos, (T, 1))
    L = np.ones((T, model.nu)) if L is None else L
    return MuscleActivity(100.0, drive, drive, np.zeros_like(drive), drive, L, q, q, np.zeros((T, model.m.nv)), np.zeros((T, model.m.nv)),
                          np.zeros(T))


@needs_bank
def test_low_rank_matches_exact(setup):
    model, _, engine = setup
    L = np.linspace(0.9, 1.1, 150)[:, None] * np.ones((1, model.nu))      # moving fibre length: bin interpolation
    act = _activity(model, 0.4, L=L)
    fast = engine.generate(act, np.random.default_rng(5)).emg
    exact = engine.generate(act, np.random.default_rng(5), exact=True).emg
    assert np.linalg.norm(fast - exact) / np.linalg.norm(exact) < 0.05


@needs_bank
def test_emg_scales_with_drive(setup):
    model, _, engine = setup
    rms = [np.sqrt((engine.generate(_activity(model, lv, T=200), np.random.default_rng(0)).emg ** 2).mean())
           for lv in (0.1, 0.5, 1.0)]
    assert rms[0] < rms[1] < rms[2]


@needs_bank
@pytest.mark.parametrize("muscle", ["EDC3_r", "ECU_r", "PL_r"])
def test_engines_agree_where_a_muscle_is(setup, muscle):
    """Both engines see an anchored muscle at the same place around the forearm (shared electrode frame)."""
    model, ring, engine = setup
    analytic = AnalyticEngine()
    analytic.prepare(model, mg.Subject(), ring)
    angles = np.radians([e.angle_deg for e in ring.electrodes])

    def peak(e):
        rms = np.sqrt((e.generate(_activity(model, 0.5, muscle, T=200), np.random.default_rng(0)).emg ** 2).mean(0))
        w = np.maximum(rms - np.median(rms), 0)
        return np.angle((w * np.exp(1j * angles)).sum())

    assert abs(np.degrees(np.angle(np.exp(1j * (peak(engine) - peak(analytic)))))) < 30


@needs_bank
def test_electrodes_beyond_the_grid_are_rejected():
    with pytest.raises(ValueError, match="outside the BioMime grid"):
        mg.Simulator(mg.Electrodes([mg.Electrode(90, 20.0)]), engine="biomime", muap_bank=BANK)


@needs_bank
@pytest.mark.parametrize("preset", ["ring", "myo", "otto_bock"])
def test_presets_fit_the_grid(preset):
    BioMimeEngine(BANK, units_per_muscle=8).prepare(ForearmModel(), mg.Subject(), mg.Electrodes.preset(preset))
