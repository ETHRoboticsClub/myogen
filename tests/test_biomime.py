"""Checks for the BioMime engine. The synthesis tests need the MUAP bank (myogen download muap-bank)."""
import numpy as np
import pytest

from myogen.emg.biomime.anatomy import GridGeometry
from myogen.emg.biomime.electrodes import Channel, interp_matrix, skin_filter
from myogen.emg.biomime.mupool import FuglevandPool, LIFPool


def test_interp_on_node_and_bipolar_zero_sum():
    g = GridGeometry()
    node = Channel([(5 / 32, float(g.z0_frac + 3 / 9 * (g.z1_frac - g.z0_frac)), 1.0)])
    W, out = interp_matrix([node], g)
    assert out == 0 and np.isclose(W[0, 3 * 32 + 5], 1.0) and np.isclose(W.sum(), 1.0)
    bip = Channel([(0.3, 0.2, 1.0), (0.3, 0.25, -1.0)])
    W, _ = interp_matrix([bip], g)
    assert np.isclose(W.sum(), 0.0)


def test_skin_filter_attenuates_and_preserves_reference():
    g = GridGeometry()
    assert np.allclose(skin_filter(g, 0.003), np.eye(320))
    B = skin_filter(g, 0.008)
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


@pytest.fixture(scope="module")
def synth():
    from pathlib import Path

    from myogen.body.model import shared_model
    from myogen.config import load_config
    from myogen.emg.biomime.bank import MuapBank
    from myogen.emg.biomime.synth import BioMimeEMG
    from myogen.subject import sample_session, sample_subject
    cfg = load_config()                       # ./myogen.yml in the repository (paths.data: data)
    if not Path(cfg["paths"]["muap_bank"]).exists():
        pytest.skip("MUAP bank not available (myogen download muap-bank)")
    m, d, anat = shared_model()
    rng = np.random.default_rng(1)
    sub = sample_subject(rng, m.nu, anat.extrinsic, cfg["electrodes"]["myo"], device="myo")
    return BioMimeEMG(MuapBank(cfg["paths"]["muap_bank"]), sub, sample_session(rng, sub), anat, units_per_muscle=24, keep_templates=True), m


def test_low_rank_matches_exact(synth):
    syn, m = synth
    T = 150
    u = np.full((T, m.nu), 0.4)
    L = np.linspace(0.9, 1.1, T)[:, None] * np.ones((1, m.nu))      # moving fibre length: bin interpolation
    fast = syn.synthesize(u, L, 100, np.random.default_rng(5))
    exact = syn.synthesize(u, L, 100, np.random.default_rng(5), exact=True)
    err = np.linalg.norm(fast - exact) / np.linalg.norm(exact)
    assert err < 0.05, err


def test_emg_scales_with_drive(synth):
    syn, m = synth
    rms = [np.sqrt((syn.synthesize(np.full((200, m.nu), lv), np.ones((200, m.nu)), 100,
                                   np.random.default_rng(0)) ** 2).mean()) for lv in (0.1, 0.5, 1.0)]
    assert rms[0] < rms[1] < rms[2]
