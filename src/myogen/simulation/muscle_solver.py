from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from scipy import signal
from scipy.optimize import lsq_linear

from ..inputs.subject import Subject
from ..params import ControlParams
from .kinematics import JointTrajectory
from .musculoskeletal import ForearmModel

log = logging.getLogger(__name__)


@dataclass
class MuscleActivity:
    """What the muscles did, per control tick (T ticks at fs). The input of the EMG step."""
    fs: float
    excitation: np.ndarray    # (T, nu) neural excitation in [0, 1]
    drive: np.ndarray         # (T, nu) neural drive the EMG follows: smoothed excitation (+ Effort), in [0, 1]
    effort: np.ndarray        # (T, nu) torque-neutral co-activation added to the drive (Effort)
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


class MuscleSolver:
    """JointTrajectory -> MuscleActivity: the muscle excitations that make the model follow the trajectory, at the
    mechanical minimum (computed muscle control, as OpenSim's CMC). Every control tick:
      1. the joint accelerations that track the reference (feed-forward + PD feedback on the error),
      2. the joint torques for them (inverse dynamics),
      3. the muscle activations that produce those torques with the least effort (bounded least squares; small
         reserve torques stand in for what the muscles cannot do),
      4. the excitations that reach those activations (inverse activation dynamics),
    then the model steps forward with those excitations, so the motion is the one the muscles produce."""

    def __init__(self, model: ForearmModel, subject: Subject, params: ControlParams | None = None) -> None:
        """subject: its recruitment cost per muscle, its armrest (support) and the run's physics."""
        m = model.m
        self.p = p = params or ControlParams()
        self.model, self.subject, self.m, self.d, self.fs = model, subject, m, model.d, p.fs
        self.steps_per_tick = int(round(1 / (self.fs * m.opt.timestep)))
        self.dt = m.opt.timestep * self.steps_per_tick
        cost = subject.recruitment_cost
        self.cost = np.ones(m.nu) if cost is None else np.asarray(cost)
        self.support = model.support(subject)
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

    def solve(self, trajectory: JointTrajectory, external_torque: np.ndarray | None = None,
              on_tick: Callable[[int, int], None] | None = None) -> MuscleActivity:
        """The muscle activity that moves the model along the trajectory, from its first posture.
        external_torque: (T, nv) torques on the joints besides the muscles (Nm), e.g. a grip; on_tick(k, T) is
        called after every tick."""
        q_ref, m, d = trajectory.q_ref, self.m, self.d
        T = len(q_ref)
        self.model.set_environment(self.subject, q_ref[0], trajectory.q_rest)
        fibre_rest = self.model.fibre_lengths_at(trajectory.q_rest)
        load_torque = np.zeros((T, m.nv)) if external_torque is None else external_torque
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
                u, reserve, out["residual"][k] = self._excitations(q_ref[k], qd_ref[k], qdd_ref[k], load_torque[k])
            except np.linalg.LinAlgError:
                out["recovered"][k] = True
                self._recover(q_ref[k], act_ok, k)
                u, reserve, out["residual"][k] = self._excitations(q_ref[k], qd_ref[k], qdd_ref[k], load_torque[k])
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
        out["effort"] = np.zeros((T, m.nu))
        out["drive"] = self._neural_drive(out["excitation"])
        return MuscleActivity(fs=self.fs, **out)

    def _excitations(self, q_ref: np.ndarray, qd_ref: np.ndarray, qdd_ref: np.ndarray, load: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, float]:
        """(excitation, reserve torque, residual) for the next tick (steps 1-4); applies the external torques, the
        armrest's support and the reserve torques to the model."""
        m, d = self.m, self.d

        # Desired acceleration: feed-forward + PD feedback on the reference.
        d.qacc[:] = qdd_ref + self.kp * (q_ref - d.qpos) + self.kd * (qd_ref - d.qvel)

        # Joint torques for it: inverse dynamics (also refreshes moment arms, muscle lengths and M).
        mujoco.mj_inverse(m, d)
        mujoco.mju_sparse2dense(self.R, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        gain, bias = self._muscle_gain_bias()
        load = load + self.support * (d.qfrc_bias - self.R.T @ bias - d.qfrc_spring)

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

    def _neural_drive(self, excitation: np.ndarray) -> np.ndarray:
        """(T, nu) drive the EMG follows: smoothed excitation (the solver's fast corrections are a simulation
        artifact; neural drive is < ~5 Hz)."""
        drive_lp = signal.butter(2, self.p.drive_lowpass_hz, fs=self.fs, output="sos")
        smooth = signal.sosfiltfilt(drive_lp, excitation, axis=0) if len(excitation) > 15 else excitation
        return np.clip(smooth, 0, 1)

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
