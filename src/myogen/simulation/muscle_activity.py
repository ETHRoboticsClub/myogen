from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy import signal
from scipy.optimize import lsq_linear

from ..params import ControlParams, EffortParams
from .musculoskeletal import HAND_JOINTS

if TYPE_CHECKING:
    from ..inputs.subject import Subject
    from .musculoskeletal import ForearmModel

log = logging.getLogger(__name__)


@dataclass
class MuscleActivity:
    """What the muscles did, per control tick (T ticks at fs). The input of the EMG step."""
    fs: float
    excitation: np.ndarray    # (T, nu) neural excitation in [0, 1]
    drive: np.ndarray         # (T, nu) neural drive the EMG follows: smoothed excitation + effort, in [0, 1]
    effort: np.ndarray        # (T, nu) torque-neutral co-activation added to the drive
    activation: np.ndarray    # (T, nu)
    fibre_length: np.ndarray  # (T, nu) relative to the relaxed posture (1 = relaxed)
    qpos: np.ndarray          # (T, nq) joint state at each tick, before stepping (muscle geometry for the EMG)
    q: np.ndarray             # (T, nq) joint angles after each tick
    qd: np.ndarray            # (T, nv) joint velocities after each tick
    reserve: np.ndarray       # (T, nv) reserve torques (Nm)
    residual: np.ndarray      # (T,) torque residual of the muscle solution
    recovered: np.ndarray | None = None   # (T,) bool: the simulation diverged and was reset to the reference here

    def __getitem__(self, s: slice) -> MuscleActivity:
        """The ticks in slice s (all arrays sliced along time)."""
        return MuscleActivity(self.fs, *(None if getattr(self, f.name) is None else getattr(self, f.name)[s]
                                         for f in fields(self)[1:]))


# -- the mechanical minimum: activations that reproduce the motion (computed muscle control) -----------------------

class MuscleController:
    def __init__(self, model: ForearmModel, muscle_cost: list[float] | np.ndarray | None = None,
                 params: ControlParams | None = None) -> None:
        """muscle_cost: (nu,) relative recruitment cost (None = equal)."""
        m = model.m
        self.p = p = params or ControlParams()
        self.model, self.m, self.d, self.fs = model, m, model.d, p.fs
        self.steps_per_tick = int(round(1 / (self.fs * m.opt.timestep)))
        self.dt = m.opt.timestep * self.steps_per_tick
        self.cost = np.ones(m.nu) if muscle_cost is None else np.asarray(muscle_cost)
        self.tau_act, self.tau_deact = m.actuator_dynprm[:, 0], m.actuator_dynprm[:, 1]

        # Reserve torques: what the muscles cannot do (the thumb lacks its intrinsic muscles: cheaper, larger ones).
        self.reserve_w = np.full(m.nv, p.reserve_cost)
        self.reserve_lim = np.full(m.nv, p.reserve_max_nm)

        for name in p.weak_joints:
            dof = m.jnt_dofadr[m.joint(name).id]
            self.reserve_w[dof], self.reserve_lim[dof] = p.weak_reserve_cost, p.weak_reserve_max_nm

        omega = 2 * np.pi * p.pd_bandwidth_hz
        self.kp = omega ** 2
        self.kd = 2 * p.pd_damping * omega

        self.R = np.zeros((m.nu, m.nv))
        self.M = np.zeros((m.nv, m.nv))

    def track(self, q_ref: np.ndarray, load_torque: np.ndarray, support: np.ndarray, fibre_rest: np.ndarray,
              effort: np.ndarray | None = None, on_tick: Callable[[int, int], None] | None = None) -> MuscleActivity:
        """Drive the model along q_ref (T, nq), starting at q_ref[0].

        load_torque: (T, nv) external torques (Nm); support: (nv,) fraction of the static torque carried by an
        armrest; fibre_rest: (nu,) fibre lengths at the relaxed posture; effort: (T, nu) co-activation added to the
        neural drive (not to the mechanics). on_tick(k, T) is called after every tick.
        """
        m, d, T = self.m, self.d, len(q_ref)
        qd_ref = np.gradient(q_ref, 1 / self.fs, axis=0)
        qdd_ref = np.gradient(qd_ref, 1 / self.fs, axis=0)
        self.model.set_pose(q_ref[0])
        self.a_prev = np.zeros(m.nu)
        out = {k: np.zeros((T, n)) for k, n in [("excitation", m.nu), ("activation", m.nu), ("fibre_length", m.nu),
                                                ("qpos", m.nq), ("q", m.nq), ("qd", m.nv), ("reserve", m.nv)]}
        out["residual"], out["recovered"] = np.zeros(T), np.zeros(T, bool)
        act_ok = np.zeros(m.nu)
        for k in range(T):
            # Excitations for this tick (restart from the reference if the simulation blew up).
            if self._diverged():
                out["recovered"][k] = True
                self._recover(q_ref[k], act_ok, k)
            act_ok = d.act.copy()
            try:
                u, reserve, out["residual"][k] = self._control(q_ref[k], qd_ref[k], qdd_ref[k], load_torque[k],
                                                               support)
            except np.linalg.LinAlgError:
                out["recovered"][k] = True
                self._recover(q_ref[k], act_ok, k)
                u, reserve, out["residual"][k] = self._control(q_ref[k], qd_ref[k], qdd_ref[k], load_torque[k],
                                                               support)
            # Step the model forward with them: the resulting motion is physically consistent.
            out["qpos"][k] = d.qpos
            d.ctrl[:] = u
            for _ in range(self.steps_per_tick):
                mujoco.mj_step(m, d)
            out["excitation"][k], out["reserve"][k] = u, reserve
            out["q"][k], out["qd"][k], out["activation"][k] = d.qpos, d.qvel, d.act
            out["fibre_length"][k] = self.model.fibre_lengths()
            if on_tick is not None:
                on_tick(k, T)
        out["fibre_length"] /= fibre_rest
        out["effort"] = np.zeros((T, m.nu)) if effort is None else np.asarray(effort, float)
        out["drive"] = self._neural_drive(out["excitation"], out["effort"])
        return MuscleActivity(fs=self.fs, **out)

    def _control(self, q_ref: np.ndarray, qd_ref: np.ndarray, qdd_ref: np.ndarray, load: np.ndarray,
                 support: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(excitation, reserve torque, residual) for the next tick; applies load + support + reserve torques."""
        m, d = self.m, self.d

        # Desired acceleration: feed-forward + PD feedback on the reference.
        d.qacc[:] = qdd_ref + self.kp * (q_ref - d.qpos) + self.kd * (qd_ref - d.qvel)

        # Joint torques for it: inverse dynamics (also refreshes moment arms, muscle lengths and M).
        mujoco.mj_inverse(m, d)
        mujoco.mju_sparse2dense(self.R, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        gain, bias = self._muscle_gain_bias()
        load = load + support * (d.qfrc_bias - self.R.T @ bias - d.qfrc_spring)

        # Muscle activations that produce those torques.
        a_des, reserve, residual = self._solve_activation(d.qfrc_inverse - load, gain, bias)
        d.qfrc_applied[:] = load + reserve

        # Excitations that reach those activations.
        return self._excitation_for(a_des), reserve, residual

    def _muscle_gain_bias(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-muscle (gain, bias) with actuator_force = gain * act + bias at the current state."""
        m, d = self.m, self.d
        gain, bias = np.empty(m.nu), np.empty(m.nu)
        for a in range(m.nu):
            L, V = d.actuator_length[a], d.actuator_velocity[a]
            gain[a] = mujoco.mju_muscleGain(L, V, m.actuator_lengthrange[a], m.actuator_acc0[a],
                                            m.actuator_gainprm[a, :9])
            bias[a] = mujoco.mju_muscleBias(L, m.actuator_lengthrange[a], m.actuator_acc0[a],
                                            m.actuator_biasprm[a, :9])
        return gain, bias

    def _solve_activation(
            self, tau: np.ndarray, gain: np.ndarray, bias: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, float]:
        """(activations (nu,), reserve torques (nv,) in Nm, residual). Bounded least squares:
        min ||R^T (gain * a + bias) + reserve - tau||^2 + effort ||a||^2 + smoothness ||a - leak a_prev||^2
        + reserve cost ||reserve||^2, with torque errors scaled to acceleration errors."""
        nu, nv = self.m.nu, self.m.nv
        # Torque errors as acceleration errors, so light finger joints are tracked as tightly as the wrist.
        mujoco.mj_fullM(self.m, self.d, self.M)
        scale = np.diag(self.M) * self.p.acc_scale
        A_tau = np.hstack([(self.R.T * gain) / scale[:, None], np.eye(nv)])
        y_tau = (tau - self.R.T @ bias) / scale
        w, ws = np.sqrt(self.p.effort_weight) * self.cost, np.sqrt(self.p.smooth_weight)
        A_reg = np.zeros((2 * nu + nv, nu + nv))
        A_reg[:nu, :nu] = np.diag(w)
        A_reg[nu:2 * nu, :nu] = np.eye(nu) * ws
        A_reg[2 * nu:, nu:] = np.diag(self.reserve_w)
        A = np.vstack([A_tau, A_reg])
        y = np.concatenate([y_tau, np.zeros(nu), ws * self.p.smooth_leak * self.a_prev, np.zeros(nv)])
        rmax = self.reserve_lim / scale
        res = lsq_linear(A, y, bounds=(np.r_[np.zeros(nu), -rmax], np.r_[np.ones(nu), rmax]), method="bvls")
        self.a_prev = res.x[:nu]
        residual = float(np.linalg.norm(A_tau @ res.x - y_tau) / np.sqrt(nv))
        return res.x[:nu], res.x[nu:] * scale, residual

    def _excitation_for(self, a_des: np.ndarray) -> np.ndarray:
        """Invert MuJoCo's first-order muscle activation dynamics over `horizon_ticks` control ticks."""
        act = self.d.act
        tau = np.where(a_des > act, self.tau_act * (0.5 + 1.5 * act), self.tau_deact / (0.5 + 1.5 * act))
        k = 1 - np.exp(-self.p.horizon_ticks * self.dt / tau)
        return np.clip(act + (a_des - act) / k, 0.0, 1.0)

    def _neural_drive(self, excitation: np.ndarray, effort: np.ndarray) -> np.ndarray:
        """(T, nu) drive the EMG follows: smoothed excitation (the controller's fast corrections are a simulation
        artifact; neural drive is < ~5 Hz) plus the effort."""
        drive_lp = signal.butter(2, self.p.drive_lowpass_hz, fs=self.fs, output="sos")
        smooth = signal.sosfiltfilt(drive_lp, excitation, axis=0) if len(excitation) > 15 else excitation
        return np.clip(np.clip(smooth, 0, 1) + effort, 0, 1)

    def _diverged(self) -> bool:
        d = self.d
        return not (np.isfinite(d.qpos).all() and np.isfinite(d.act).all() and np.isfinite(d.qvel).all()
                    and np.abs(d.qvel).max() < self.p.max_speed)

    def _recover(self, q: np.ndarray, act: np.ndarray, k: int) -> None:
        """Restart from the reference pose at rest, keeping the last sane activations."""
        log.warning(f"simulation diverged at tick {k} ({k / self.fs:.2f} s): reset to the reference pose")
        self.model.set_pose(q)
        self.d.act[:] = act
        mujoco.mj_forward(self.m, self.d)
        self.a_prev = np.zeros(self.m.nu)


# -- effort beyond the mechanical minimum: co-activation and grip -------------------------------------------------

def effort(model: ForearmModel, q_ref: np.ndarray, q_rest: np.ndarray, subject: Subject, rng: np.random.Generator,
           params: EffortParams | None = None, fs: float = 100.0, rest: np.ndarray | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """(co-activation (T, nu), grip load (T,)). People drive their muscles harder than the mechanical minimum.

    A joint is engaged while it moves or is held away from the relaxed posture, in the direction it moves or is held.
    Fingers are not independent: a finger's engagement spreads to the others (EffortParams.finger_coupling).
    Each muscle gets the effort of the engaged joints it acts on, in proportion to its moment arm relative to the
    joint's strongest muscle: in full if it pulls the joint that way (agonist), a fraction
    (Subject.antagonist_ratio) if it pulls the other way. A relaxed hand is nearly electrically silent, as in real
    recordings: where the motion says the hand is relaxed (rest: (T,) bool), the effort falls to
    EffortParams.rest_effort of its value.
    """
    p = params or EffortParams()
    engaged, moving = joint_engagement(model, q_ref, q_rest, p, fs)
    weight = joint_weights(model, p.joint_effort)
    engaged = np.clip(couple_fingers(model, engaged, p.finger_coupling) * weight, -1, 1)
    moving = np.clip(couple_fingers(model, moving, p.finger_coupling) * weight, -p.speed_max, p.speed_max)
    involvement = muscle_involvement(model, q_rest)
    # Co-activation level: the subject's baseline with random episodes, plus tone while engaged and moving.
    level = _episodes(rng, len(q_ref), fs, base=subject.cocontraction, high=p.cocontraction_peak,
                      p_episode=p.cocontraction_episode_prob, duration_s=p.episode_s,
                      tau=p.episode_smoothing_s) + subject.hold_tone
    r = subject.antagonist_ratio
    co_activation = (level[:, None] * per_muscle(engaged, involvement, r)
                     + subject.move_cocontraction * per_muscle(moving, involvement, r))
    if rest is not None and p.rest_effort < 1:
        lp = signal.butter(2, p.speed_lowpass_hz, fs=fs, output="sos")
        relaxed = np.clip(signal.sosfiltfilt(lp, np.asarray(rest, float)), 0, 1)
        co_activation *= (1 - (1 - p.rest_effort) * relaxed)[:, None]
    return co_activation, grip_load(model, q_ref, subject, rng, p, fs)


def joint_weights(model: ForearmModel, joint_effort: dict[str, float]) -> np.ndarray:
    """(nv,) effort weight of each joint: the longest prefix of its name in joint_effort (1 if none)."""
    w = np.ones(model.m.nv)
    for j in range(model.m.njnt):
        name = model.m.joint(j).name
        match = max((k for k in joint_effort if name.startswith(k)), key=len, default=None)
        if match is not None:
            w[model.m.jnt_dofadr[j]] = joint_effort[match]
    return w


def couple_fingers(model: ForearmModel, joint_level: np.ndarray, coupling: float) -> np.ndarray:
    """(T, nv) joint levels with each finger's (2-5) flexion levels spread to the same joints of the other fingers,
    weighted by coupling ** (distance between the fingers): the enslaving of the multi-finger muscles."""
    if coupling <= 0:
        return joint_level
    out = joint_level.copy()
    for joint in ("mcp{}_flexion_r", "pm{}_flexion_r", "md{}_flexion_r"):
        dofs = [model.m.jnt_dofadr[model.m.joint(joint.format(f)).id] for f in "2345"]
        for i, a in enumerate(dofs):
            for k, b in enumerate(dofs):
                if k != i:
                    out[:, a] += coupling ** abs(i - k) * joint_level[:, b]
    return out


def joint_engagement(model: ForearmModel, q_ref: np.ndarray, q_rest: np.ndarray, p: EffortParams, fs: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(engaged (T, nv) in [-1, 1], moving (T, nv)): how far each joint is held from rest and how fast it moves,
    signed by direction (the sign of the joint angle change it calls for). Moving back towards the relaxed posture
    is mostly letting go: it counts EffortParams.return_effort as much as moving away from it."""
    span = np.maximum(model.qmax - model.qmin, 1e-6)
    u, u_rest = q_ref / span, q_rest / span                     # joint angles as fractions of their range
    lp = signal.butter(2, p.speed_lowpass_hz, fs=fs, output="sos")
    velocity = np.gradient(u, axis=0) * fs
    speed = signal.sosfiltfilt(lp, np.abs(velocity), axis=0)
    direction = signal.sosfiltfilt(lp, velocity, axis=0)
    outward = np.where(direction * (u - u_rest) >= 0, 1.0, p.return_effort)
    moving = np.sign(direction) * np.clip(speed - p.speed_deadzone, 0, p.speed_max) * outward
    away = np.sign(u - u_rest) * np.clip(np.abs(u - u_rest) - p.posture_deadzone, 0, None) / p.posture_engaged
    engaged = np.clip(signal.sosfiltfilt(lp, np.clip(away + moving / p.speed_engaged, -1, 1), axis=0), -1, 1)
    return engaged, moving


def muscle_involvement(model: ForearmModel, q_rest: np.ndarray) -> np.ndarray:
    """(nu, nv) each muscle's moment arm relative to the strongest muscle of each joint (one dof per joint), signed
    by the direction the muscle turns the joint (positive: towards larger angles)."""
    R = -model.moment_arms(q_rest)                  # muscles pull: their force is negative in MuJoCo
    return R / np.maximum(np.abs(R).max(0), 1e-9)


def per_muscle(joint_level: np.ndarray, involvement: np.ndarray, antagonist_ratio: float = 1.0) -> np.ndarray:
    """(T, nu) the largest level among the joints each muscle acts on, weighted by its involvement: in full for the
    muscles that turn the joint in the level's direction, antagonist_ratio for the others (1: direction ignored)."""
    out = np.zeros((len(joint_level), involvement.shape[0]))
    for j in range(involvement.shape[1]):
        agonist = np.sign(joint_level[:, j:j + 1]) * np.sign(involvement[:, j]) > 0
        weight = np.abs(involvement[:, j]) * np.where(agonist, 1.0, antagonist_ratio)
        out = np.maximum(out, np.abs(joint_level[:, j:j + 1]) * weight)
    return out


def grip_load(model: ForearmModel, q_ref: np.ndarray, subject: Subject, rng: np.random.Generator,
              p: EffortParams, fs: float) -> np.ndarray:
    """(T,) random squeezing episodes in [0, 1], only while the hand is closed."""
    closure = finger_closure(q_ref[:, model.hand_qpos], p.pip_closed_rad).mean(1)
    closed = 1 / (1 + np.exp(-(closure - p.closed_hand) / p.closed_hand_width))
    return _episodes(rng, len(q_ref), fs, base=0.0, high=1.0, p_episode=subject.grip_load, duration_s=p.episode_s,
                     tau=p.episode_smoothing_s) * closed


def finger_closure(hand: np.ndarray, pip_closed_rad: float = EffortParams.pip_closed_rad) -> np.ndarray:
    """(T, 4) closure of fingers 2-5 in [0, 1], from the PIP angles."""
    return np.clip(hand[:, [HAND_JOINTS.index(f"pm{f}_flexion_r") for f in "2345"]] / pip_closed_rad, 0, 1)


def _episodes(rng: np.random.Generator, T: int, fs: float, base: float, high: float, p_episode: float,
              duration_s: tuple[float, float], tau: float) -> np.ndarray:
    """(T,) slowly varying level at `base` with random episodes up to `high`, zero-phase smoothed with tau."""
    x = np.full(T, float(base))
    i = 0
    while i < T:
        n = int(rng.uniform(*duration_s) * fs)
        if rng.random() < p_episode:
            x[i:i + n] = rng.uniform(base, high)
        i += n
    a = np.exp(-1 / (tau * fs))
    for _ in range(2):                         # first-order low-pass forward, then backward
        y, acc = np.empty_like(x), x[0]
        for k in range(T):
            acc = a * acc + (1 - a) * x[k]
            y[k] = acc
        x = y[::-1]
    return x
