import numpy as np
import pytest

from myogen.params import VolumeConductorParams
from myogen.simulation.volume_conductor import LayeredCylinder, MUAPTable


def test_half_space_limit():
    """One conductivity everywhere and a large radius: a shallow source under the insulated skin gives the half-space
    potential I / (2 pi sigma r), up to a constant (the model leaves out the zero axial wavenumber, which a fibre's
    sources never excite): compare differences between points."""
    s = 0.3
    p = VolumeConductorParams(sigma_muscle_radial=s, sigma_muscle_axial=s, sigma_fat=s, sigma_skin=s, sigma_bone=s,
                              bone_radius_mm=10, angular_orders=160)
    cyl = LayeredCylinder(p, 0.060, 0.003)
    for depth in (0.004, 0.008):
        h = cyl.point_potential(cyl.d - depth, np.array([0.0]), 1024, 0.00025)[0]
        expected = 1 / (2 * np.pi * s * depth) - 1 / (2 * np.pi * s * np.hypot(depth, 0.01))
        assert h[0] - h[40] == pytest.approx(expected, rel=0.05)          # over the source vs 10 mm along


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    import myogen.simulation.volume_conductor as vc
    vc.CACHE_DIR = tmp_path_factory.mktemp("muap")
    return MUAPTable(VolumeConductorParams(depths_mm=(0.5, 2, 4, 8, 12, 20, 28)), 0.036, 0.005, 2000.0)


def test_table_lookup(table):
    """On the grid the lookup returns the stored weights; beyond the table's distance it is zero."""
    i, j, k = 2, 10, 30
    w = table.weights_at(np.array(table.depth[i]), np.array(table.theta[j]), np.array(table.dz[k]))
    assert np.allclose(w, table.weights[i, j, k])
    assert not table.weights_at(np.array(0.004), np.array(0.0), np.array(0.5)).any()


def test_bipolar_selectivity(table):
    """A bipolar pair (10 mm along the fibres) loses a deep unit faster than a single pole does, and a unit to the
    side faster than one beneath."""
    def rms(depth, theta, bipolar):
        w = table.weights_at(np.array(depth), np.array(theta), np.array(0.02))
        if bipolar:
            w = w - table.weights_at(np.array(depth), np.array(theta), np.array(0.03))
        return np.sqrt(np.mean((w @ table.basis) ** 2))
    for bipolar in (False, True):
        assert rms(0.02, 0.0, bipolar) < rms(0.002, 0.0, bipolar)
        assert rms(0.004, 0.6, bipolar) < rms(0.004, 0.0, bipolar)
    assert rms(0.02, 0, True) / rms(0.002, 0, True) < rms(0.02, 0, False) / rms(0.002, 0, False)
