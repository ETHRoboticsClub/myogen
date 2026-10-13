"""MuscleController: the muscle excitations that make the model follow a joint trajectory (computed muscle control).

Each control tick (100 Hz):
  1. desired acceleration = feed-forward + PD feedback on the reference
  2. joint torque         = MuJoCo inverse dynamics (gravity, inertia, passive forces, minus external loads)
  3. activations a        = bounded least squares
                            min ||W (R^T (gain * a + bias) - tau)||^2 + lam ||a||^2 + smoothness
                            plus heavily penalised reserve torques for what the muscles cannot do (the thumb
                            lacks intrinsic muscles; reserves stand in for them without producing EMG)
  4. excitation u         = inverse activation dynamics, so the activation reaches a over `horizon` ticks
The model is then stepped forward with u: q in the output is the physically consistent motion.

This is the mechanical minimum. People drive their muscles harder (stiffening, co-contraction): that effort is
added to the neural drive as torque-neutral co-activation, so it reaches the EMG without changing the motion.
"""
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from scipy import signal
from scipy.optimize import lsq_linear


@dataclass
class MuscleActivity:
    """What the muscles did, per control tick (T ticks at fs). The input of the EMG engines."""
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

    def __getitem__(self, s):
        """The ticks in slice s (all arrays sliced along time)."""
        return MuscleActivity(self.fs, *(getattr(self, f.name)[s] for f in fields(self)[1:]))


class MuscleController:
    omega = 2 * np.pi * 8.0          # PD tracking bandwidth
    zeta = 1.0
    lam = 3e-3                       # effort / co-contraction regularisation weight
    smooth = 1.0                     # penalty on activation change between ticks (smooth neural drive)
    smooth_leak = 0.9                # pull towards leak * a_prev: unneeded drive decays in ~100 ms
    horizon = 3                      # ticks over which the activation dynamics are inverted
    acc_scale = 200.0                # rad/s^2 of acceleration error that costs as much as lam
    reserve_cost, reserve_max = 1.0, 0.5               # reserve torques: weight, Nm per joint
    weak_joints = ("cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r")   # thumb: no intrinsics
    weak_reserve_cost, weak_reserve_max = 0.1, 3.0

    def __init__(self, model, muscle_cost=None, fs=100.0):
        """model: a ForearmModel; muscle_cost: (nu,) relative recruitment cost (None = equal)."""
        m = model.m
        self.model, self.m, self.d, self.fs = model, m, model.d, fs
        self.steps_per_tick = int(round(1 / (fs * m.opt.timestep)))
        self.dt = m.opt.timestep * self.steps_per_tick
        self.cost = np.ones(m.nu) if muscle_cost is None else np.asarray(muscle_cost)
        self.tau_act, self.tau_deact = m.actuator_dynprm[:, 0], m.actuator_dynprm[:, 1]
        self.reserve_w = np.full(m.nv, self.reserve_cost)
        self.reserve_lim = np.full(m.nv, self.reserve_max)
        for name in self.weak_joints:
            dof = m.jnt_dofadr[m.joint(name).id]
            self.reserve_w[dof], self.reserve_lim[dof] = self.weak_reserve_cost, self.weak_reserve_max
        self.kp, self.kd = self.omega ** 2, 2 * self.zeta * self.omega
        self.R = np.zeros((m.nu, m.nv))
        self.M = np.zeros((m.nv, m.nv))

    def track(self, q_ref, load_torque, support, fibre_rest, effort=None, on_tick=None):
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
        out["residual"] = np.zeros(T)
        for k in range(T):
            out["qpos"][k] = d.qpos
            u, reserve, out["residual"][k] = self._control(q_ref[k], qd_ref[k], qdd_ref[k], load_torque[k], support)
            d.ctrl[:] = u
            for _ in range(self.steps_per_tick):
                mujoco.mj_step(m, d)
            out["excitation"][k], out["reserve"][k] = u, reserve
            out["q"][k], out["qd"][k], out["activation"][k] = d.qpos, d.qvel, d.act
            out["fibre_length"][k] = self.model.fibre_lengths()
            if on_tick is not None:
                on_tick(k, T)
        out["fibre_length"] /= fibre_rest
        exc = out["excitation"]
        # Neural drive is smooth (< ~5 Hz): the controller's fast corrections are a simulation artifact.
        smooth = signal.sosfiltfilt(signal.butter(2, 4.0, fs=self.fs, output="sos"), exc, axis=0) if T > 15 else exc
        out["effort"] = np.zeros((T, m.nu)) if effort is None else np.asarray(effort, float)
        out["drive"] = np.clip(np.clip(smooth, 0, 1) + out["effort"], 0, 1)
        return MuscleActivity(fs=self.fs, **out)

    def _control(self, q_ref, qd_ref, qdd_ref, load, support):
        """(excitation, reserve torque, residual) for the next tick; applies load + support + reserve torques."""
        m, d = self.m, self.d
        d.qacc[:] = qdd_ref + self.kp * (q_ref - d.qpos) + self.kd * (qd_ref - d.qvel)
        mujoco.mj_inverse(m, d)          # also refreshes moments, muscle lengths / velocities and M
        mujoco.mju_sparse2dense(self.R, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
        gain, bias = self._muscle_gain_bias()
        load = load + support * (d.qfrc_bias - self.R.T @ bias - d.qfrc_spring)
        a_des, reserve, residual = self._solve_activation(d.qfrc_inverse - load, gain, bias)
        d.qfrc_applied[:] = load + reserve
        return self._excitation_for(a_des), reserve, residual

    def _muscle_gain_bias(self):
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

    def _solve_activation(self, tau, gain, bias):
        """(activations (nu,), reserve torques (nv,) in Nm, residual)."""
        nu, nv = self.m.nu, self.m.nv
        # Torque errors as acceleration errors, so light finger joints are tracked as tightly as the wrist.
        mujoco.mj_fullM(self.m, self.d, self.M)
        scale = np.diag(self.M) * self.acc_scale
        A_tau = np.hstack([(self.R.T * gain) / scale[:, None], np.eye(nv)])
        y_tau = (tau - self.R.T @ bias) / scale
        w, ws = np.sqrt(self.lam) * self.cost, np.sqrt(self.smooth)
        A_reg = np.zeros((2 * nu + nv, nu + nv))
        A_reg[:nu, :nu] = np.diag(w)
        A_reg[nu:2 * nu, :nu] = np.eye(nu) * ws
        A_reg[2 * nu:, nu:] = np.diag(self.reserve_w)
        A = np.vstack([A_tau, A_reg])
        y = np.concatenate([y_tau, np.zeros(nu), ws * self.smooth_leak * self.a_prev, np.zeros(nv)])
        rmax = self.reserve_lim / scale
        res = lsq_linear(A, y, bounds=(np.r_[np.zeros(nu), -rmax], np.r_[np.ones(nu), rmax]), method="bvls")
        self.a_prev = res.x[:nu]
        residual = float(np.linalg.norm(A_tau @ res.x - y_tau) / np.sqrt(nv))
        return res.x[:nu], res.x[nu:] * scale, residual

    def _excitation_for(self, a_des):
        """Invert MuJoCo's first-order muscle activation dynamics over one control tick."""
        act = self.d.act
        tau = np.where(a_des > act, self.tau_act * (0.5 + 1.5 * act), self.tau_deact / (0.5 + 1.5 * act))
        k = 1 - np.exp(-self.horizon * self.dt / tau)
        return np.clip(act + (a_des - act) / k, 0.0, 1.0)
