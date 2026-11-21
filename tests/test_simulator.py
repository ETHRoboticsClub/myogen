"""The pipeline: shapes, determinism, files, electrode physics."""
import mujoco
import numpy as np
import pytest

import myogen as mg
from myogen.params import EMGParams
from myogen.simulation.effort import Effort, EffortPlan, effort
from myogen.simulation.emg import EMGModel
from myogen.simulation.geometry import ForearmGeometry
from myogen.simulation.kinematics import JointTrajectory
from myogen.simulation.muscle_solver import MuscleActivity, MuscleSolver
from myogen.simulation.musculoskeletal import ForearmModel


@pytest.fixture(scope="module")
def recording():
    return mg.Simulator(mg.Electrodes.myo()).run(mg.example_motion(2.0, fs=60))


def test_shapes_and_quality(recording):
    assert recording.emg.shape == (400, 16) and recording.fs == 200
    assert recording.labels["q"].shape == (200, 24) and recording.qc["pass"]


def test_deterministic(recording):
    again = mg.Simulator(mg.Electrodes.myo()).run(mg.example_motion(2.0, fs=60))
    assert np.array_equal(again.emg, recording.emg)


def test_batch_matches_single(recording):
    """A list of motions in worker processes gives each motion's own recording, in order."""
    batch = mg.Simulator(mg.Electrodes.myo()).run([mg.example_motion(1.0, fs=60), mg.example_motion(2.0, fs=60)],
                                                  workers=2, progress=False)
    assert len(batch) == 2 and np.array_equal(batch[1].emg, recording.emg)


def test_file_roundtrip(recording, tmp_path):
    back = mg.Recording.load(recording.save(tmp_path / "r.h5"))
    assert np.array_equal(back.emg, recording.emg) and back.electrodes == recording.electrodes
    assert back.subject == recording.subject and np.allclose(back.labels["q"], recording.labels["q"])


def test_electrode_off_the_forearm():
    with pytest.raises(ValueError, match="off the forearm"):
        mg.Simulator(mg.Electrodes([mg.Electrode(0, 40.0)]))
    amputee = mg.Subject(amputee=True, residual_length=0.5)
    with pytest.raises(ValueError, match="off the forearm"):
        mg.Simulator(mg.Electrodes([mg.Electrode(0, 20.0)]), subject=amputee)


def _rms(electrodes, muscle="EDC3_r", level=0.5):
    model = ForearmModel()
    emg = EMGModel(model, mg.Subject(), electrodes, EMGParams())
    T, nu = 200, model.nu
    drive = np.zeros((T, nu))
    drive[:, model.muscle_names.index(muscle)] = level
    q = np.tile(model.default_qpos, (T, 1))
    nv = model.m.nv
    activity = MuscleActivity(100.0, drive, drive, np.zeros_like(drive), drive, np.ones((T, nu)), q, q,
                              np.zeros((T, nv)), np.zeros((T, nv)), np.zeros(T))
    return np.sqrt((emg.generate(activity, np.random.default_rng(0)) ** 2).mean(0))


def test_bipolar_physics():
    """Over EDC (60 deg in the atlas): an electrode along the fibres sees more than one across them, a wider pole
    spacing about as much or more (EDC is superficial: 20 mm already decorrelates the poles), and the opposite side
    of the forearm much less."""
    along, across, wide, palmar = _rms(mg.Electrodes([
        mg.Electrode(60, 7, ied_mm=20, orientation_deg=0), mg.Electrode(60, 7, ied_mm=20, orientation_deg=90),
        mg.Electrode(60, 7, ied_mm=40, orientation_deg=0), mg.Electrode(-120, 7, ied_mm=20)]))
    assert along > 2 * across and wide > 0.9 * along and palmar < 0.2 * along


def test_anatomy():
    """The atlas puts the muscles the lines of action misplace where they are (supinated forearm, proximal third):
    brachioradialis radial-palmar, FCU ulnar, ECRL radial-dorsal, EDC dorsal, and superficial ones near the skin."""
    model = ForearmModel()
    emg = EMGModel(model, mg.Subject(), mg.Electrodes([mg.Electrode(0, 7)]), EMGParams())
    g = emg.geometry
    pts, owner = g.muscle_points(model.d)
    for name, (lo, hi) in {"BRD_r": (-70, 0), "FCU_r": (140, 200), "ECRL_r": (0, 45), "EDC3_r": (30, 90)}.items():
        s = pts[owner == model.muscle_names.index(name)]
        s = s[s[:, 2] < np.median(s[:, 2])]                       # the proximal half, under a band
        angle = np.degrees(np.angle(np.mean(np.exp(1j * np.arctan2(s[:, 1], s[:, 0])))))
        assert (angle - lo) % 360 <= hi - lo, (name, angle)                  # within [lo, hi], wrapping
        depth = g.skin_radius(s[:, 2]) - np.hypot(s[:, 0], s[:, 1]) - g.fat
        assert np.median(depth) < 0.012, (name, depth)


def test_motor_units():
    """Recruitment and rate coding: more drive, more units firing, faster; a bipolar pair cancels far units, so it
    sees less of the opposite side of the forearm than a monopolar electrode does."""
    ring = lambda **kw: mg.Electrodes([mg.Electrode(a, 7, ied_mm=10) for a in (90, 0, -90, 180)], **kw)
    low, high = (_rms(ring(), level=x)[0] for x in (0.05, 0.5))
    assert 0 < low < 0.5 * high
    bipolar, monopolar = _rms(ring()), _rms(ring(derivation="monopolar"))
    assert bipolar[2] / bipolar[0] < monopolar[2] / monopolar[0]
    units = mg.Simulator(ring()).emg.pool
    spikes = units.spikes(np.full((100, 48), 0.3), 100.0, np.random.default_rng(0))
    rates = np.bincount(spikes.units, minlength=units.n_units) / 1.0
    assert (rates[units.threshold > 0.36] == 0).all() and rates[units.threshold < 0.1].mean() > 10   # 5% common drive


def test_monopolar():
    """One pole per electrode; the common average reference leaves channels that sum to zero."""
    ring = lambda **kw: mg.Electrodes([mg.Electrode(a, 7, ied_mm=20) for a in (90, 0, -90, 180)], **kw)
    sim = mg.Simulator(ring(derivation="monopolar", reference="average"))
    assert sim.emg.poles.shape == (4, 1, 3)
    clean = sim.simulate(mg.example_motion(1.0, fs=60))
    assert np.allclose(clean.emg.sum(1), 0, atol=1e-4 * np.abs(clean.emg).max())


def test_effort_reaches_the_drive_not_the_mechanics():
    """Effort is torque-neutral co-activation: it raises the EMG drive and leaves the motion unchanged."""
    model = ForearmModel()
    T = 150
    q = np.tile(model.default_qpos, (T, 1))
    co = np.zeros((T, model.nu))
    co[:, model.muscle_names.index("EDC3_r")] = 0.2
    plain = MuscleSolver(model, mg.Subject()).solve(JointTrajectory(q, q[0], q[:, model.hand_qpos], warmup=0))
    effortful = Effort(model, mg.Subject()).add_to_drive(plain, EffortPlan(co, np.zeros(T)))
    assert np.array_equal(plain.q, effortful.q)
    i = model.muscle_names.index("EDC3_r")
    assert effortful.drive[:, i].min() >= 0.2 > plain.drive[:, i].max()


def test_effort_is_directional():
    """Holding the wrist flexed co-activates mainly the wrist flexors, holding it extended mainly the extensors;
    the antagonists get Subject.antagonist_ratio of the agonists' effort."""
    from myogen.params import EffortParams
    model = ForearmModel()
    T, wrist = 300, model.m.jnt_qposadr[model.m.joint("flexion_r").id]
    q_rest = model.default_qpos.copy()
    flexors, extensors = ([model.muscle_names.index(n) for n in names]
                          for names in (("FCR_r", "FCU_r"), ("ECRB_r", "ECRL_r")))
    subject = mg.Subject(antagonist_ratio=0.25)
    for angle, agonists, antagonists in ((0.6, flexors, extensors), (-0.6, extensors, flexors)):
        q = np.tile(q_rest, (T, 1))
        q[:, wrist] = angle
        co, _ = effort(model, q, q_rest, subject, np.random.default_rng(0), EffortParams(), 100.0)
        held = co[T // 2]
        assert held[agonists].min() > 2 * held[antagonists].max() > 0


def test_tone():
    """Every muscle keeps rest_tone on top of its drive."""
    T, nu = 100, 3
    drive = np.zeros((T, nu))
    drive[20:30, 0] = 0.5
    z = np.zeros((T, nu))
    activity = MuscleActivity(100.0, z, drive, z, z, np.ones((T, nu)), z, z, z, z, np.zeros(T))
    toned = Effort(ForearmModel(), mg.Subject(rest_tone=0.04)).add_to_drive(activity, EffortPlan(z, np.zeros(T))).drive
    assert np.allclose(toned[:20], 0.04) and np.allclose(toned[25, 0], 0.54)


def test_subject_build():
    """The reference preset is the default subject; strength scales the model's muscle forces (and the EMG
    amplitude, more weakly) and the forearm radius; fat adds to the radius."""
    ref = mg.Subject.preset("reference")
    assert ref.forearm_radius_mm == pytest.approx(mg.Subject().forearm_radius_mm)
    strong, weak = mg.Subject.preset("high_strength"), mg.Subject.preset("low_strength")
    assert strong.forearm_radius_mm > weak.forearm_radius_mm
    assert mg.Subject.preset("high_fat").forearm_radius_mm - ref.forearm_radius_mm == pytest.approx(4.0)
    force = lambda s: mg.Simulator("myo", s).model.m.actuator_gainprm[:, 2]   # noqa: E731
    assert force(strong) == pytest.approx(force(mg.Subject()) * 1.3)
    extrinsic = np.ones(3, bool)
    assert (strong.muscle_amplitudes(extrinsic) > 1).all() and (weak.muscle_amplitudes(extrinsic) < 1).all()


def test_subject_anatomy():
    """A subject's cross-section turned, one muscle shifted and pushed deeper, another made wider."""
    model = ForearmModel()
    n, brd, edc = model.nu, model.muscle_names.index("BRD_r"), model.muscle_names.index("EDC3_r")
    base = mg.Subject()
    shift, depth, size = np.zeros(n), np.ones(n), np.ones(n)
    shift[brd], depth[brd], size[edc] = 20.0, 1.5, 2.0
    varied = mg.Subject(section_rotation_deg=10.0, muscle_shift_deg=shift.tolist(), muscle_depth=depth.tolist(),
                        muscle_size=size.tolist())
    def points(subject, m):
        return ForearmGeometry(model, subject).anatomy.points[m]
    turn = lambda a, b: np.degrees(np.angle(np.mean(np.exp(1j * (a[:, 1] - b[:, 1])))))   # noqa: E731
    assert turn(points(varied, brd), points(base, brd)) == pytest.approx(30.0, abs=0.5)
    assert np.mean(points(varied, brd)[:, 2]) < np.mean(points(base, brd)[:, 2])          # deeper: nearer the axis
    assert turn(points(varied, edc), points(base, edc)) == pytest.approx(10.0, abs=1.0)
    assert np.std(points(varied, edc)[:, 2]) > 1.2 * np.std(points(base, edc)[:, 2])       # wider


def _winding(v: np.ndarray, f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Generalised winding number of points p around the triangle mesh (v, f): ~1 inside, ~0 outside."""
    a, b, c = (v[f[:, k]][None] - p[:, None] for k in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
    num = np.einsum("ntk,ntk->nt", a, np.cross(b, c))
    den = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
    return 2 * np.arctan2(num, den).sum(1) / (4 * np.pi)


def test_muscle_surfaces():
    """Every atlas muscle has a closed surface that wraps its volume points, at a flexed pose of a varied subject."""
    model = ForearmModel()
    emg = EMGModel(model, mg.Subject.random(3), mg.Electrodes([mg.Electrode(0, 7)]), EMGParams())
    d = mujoco.MjData(model.m)
    d.qpos[:] = model.default_qpos
    d.qpos[model.elbow_qpos] = 1.3
    d.qpos[model.m.jnt_qposadr[model.m.joint("flexion_r").id]] = 0.5
    mujoco.mj_kinematics(model.m, d)
    g = emg.geometry
    pts, owner = g.muscle_points(d)
    surfaces = g.muscle_surfaces(d)
    assert {m for m, _, _ in surfaces} == set(g.anatomy.points)
    inside = {model.muscle_names[m]: np.mean(np.abs(_winding(v, f, pts[owner == m])) > 0.5) for m, v, f in surfaces}
    assert np.median(list(inside.values())) > 0.85, inside
    assert min(inside.values()) > 0.6, inside
    assert {n for n, _, _ in g.bone_surfaces()} == {"radius", "ulna"}


def test_elbow_does_not_move_forearm_muscles():
    """The elbow angle swings the humeral origins of BRD and ECRL around the forearm frame; the atlas muscles must
    stay where they are in the forearm (and inside it)."""
    model = ForearmModel()
    emg = EMGModel(model, mg.Subject(), mg.Electrodes([mg.Electrode(0, 7)]), EMGParams())
    d = mujoco.MjData(model.m)
    d.qpos[:] = model.default_qpos
    mujoco.mj_kinematics(model.m, d)
    g, units = emg.geometry, emg.placement.atlas
    straight = g.atlas_positions(units, g.bellies(d))
    d.qpos[model.elbow_qpos] = np.radians(100)
    mujoco.mj_kinematics(model.m, d)
    flexed = g.atlas_positions(units, g.bellies(d))
    assert np.allclose(straight, flexed)
    placed, _ = emg.placement.positions(d.qpos, np.arange(emg.pool.n_units))     # as the EMG uses them
    assert np.all(np.hypot(placed[:, 0], placed[:, 1]) <= g.under_fat(placed[:, 2]) + 1e-9)


def test_viewer(recording):
    """The replay viewer builds the scene and draws frames (muscle surfaces, paths, EMG) without a browser."""
    viser = pytest.importorskip("viser")
    from myogen.tools.viewer import ReplayView
    server = viser.ViserServer(port=0, verbose=False)
    try:
        view = ReplayView(server, ForearmModel(), recording)
        assert len(view.muscles) == len(view.geometry.anatomy.meshes) and view.paths
        for t in (0.0, 0.7, 1.9):
            view._last_muscles = view._last_plot = 0.0
            view.update(t)
    finally:
        server.stop()
