"""Computed muscle control: drive the 39 MyoHand muscles so the hand follows an intent trajectory.

Each control tick:
  1. desired acceleration  = feed-forward + PD feedback on the reference
  2. required joint torque = MuJoCo inverse dynamics at the current state (gravity, inertia, passive,
                             minus any external load)
  3. muscle activations    = bounded least squares   min ||W (R^T (g*a + b) - tau)||^2 + lam ||D (a - c)||^2
                             (W = 1 / (diag(M) * acc_scale): torque error measured as acceleration error)
                             with 0 <= a <= 1, where c is the co-contraction level (c = 0: minimum effort;
                             c > 0: antagonists are pulled up together -> stiffer hand, same kinematics)
     plus reserve torques r (OpenSim-style reserve actuators, heavily penalised) for what the muscles cannot
     do. MyoHand's thumb lacks APB/FPB/AdP and cannot hold itself against gravity with muscles alone; those
     muscles are intrinsic (invisible to forearm electrodes, absent in transradial amputees), so the reserves
     stand in for them without producing EMG. Reserve usage is logged for QC.
  4. neural excitation     = activation-dynamics inversion, so that act approaches a over `horizon` ticks
     (a penalty on activation change between ticks keeps the drive smooth, as real EMG envelopes are)

The excitation u(t) is what drives the EMG (it precedes force by the activation dynamics, i.e. the
electromechanical delay), the activation a(t) is what produces force, and q(t) is the resulting
physically consistent kinematics.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .model import dense_moments, muscle_gain_bias


@dataclass
class TrackerConfig:
    ctrl_every: int = 5          # physics steps per control tick (dt=2ms -> 100 Hz control)
    omega: float = 2 * np.pi * 8.0
    zeta: float = 1.0
    lam: float = 3e-3            # effort / co-contraction regularisation weight
    smooth: float = 1.0          # penalty on activation change between ticks (smooth neural drive)
    smooth_leak: float = 0.9     # smoothness pulls towards leak * a_prev: drive no longer needed decays with
                                 # tau ~ dt / (1 - leak) = 100 ms instead of lingering (the effort term is weak)
    horizon: int = 3             # ticks over which the activation dynamics are inverted (1 = most aggressive)
    acc_scale: float = 200.0      # rad/s^2 of acceleration error that costs as much as lam
    reserve_cost: float = 1.0    # weight of reserve torques (in acc_scale units); >> muscle cost
    # Joints whose actuating muscles the model lacks get cheap reserves: the thumb (no APB/FPB/AdP in either
    # model) and, in MyoHand only, supination (no supinator/biceps; added automatically when SUP_r is absent).
    # Everywhere else muscles must do the work.
    weak_joints: tuple = ("cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r")
    weak_reserve_cost: float = 0.1
    weak_reserve_max: float = 3.0   # Nm: the missing supinator/biceps are strong
    reserve_max: float = 0.5     # Nm per joint
    muscle_cost: np.ndarray | None = None   # (nu,) per-muscle recruitment cost (subject "strategy")


class MuscleTracker:
    def __init__(self, m, d, cfg: TrackerConfig):
        self.m, self.d, self.cfg = m, d, cfg
        self.R = np.zeros((m.nu, m.nv))
        self.M = np.zeros((m.nv, m.nv))
        self.dt = m.opt.timestep * cfg.ctrl_every
        self.cost = np.ones(m.nu) if cfg.muscle_cost is None else cfg.muscle_cost
        self.tau_act = m.actuator_dynprm[:, 0]
        self.tau_deact = m.actuator_dynprm[:, 1]
        self.reserve_w = np.full(m.nv, cfg.reserve_cost)
        self.reserve_max = np.full(m.nv, cfg.reserve_max)
        weak = list(cfg.weak_joints)
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "SUP_r") < 0:
            weak.append("pro_sup_r")
        for name in weak:
            dof = m.jnt_dofadr[m.joint(name).id]
            self.reserve_w[dof] = cfg.weak_reserve_cost
            self.reserve_max[dof] = cfg.weak_reserve_max
        self.kp = cfg.omega ** 2
        self.kd = 2 * cfg.zeta * cfg.omega
        self.last_residual = 0.0
        self.a_prev = np.zeros(m.nu)

    def _torque_scale(self):
        # Express torque errors as acceleration errors (divide by joint-space inertia), so light distal
        # joints are tracked as tightly as the wrist instead of being dominated by the effort cost.
        mujoco.mj_fullM(self.m, self.d, self.M)
        return np.diag(self.M) * self.cfg.acc_scale

    def solve_activation(self, tau, cocon, gain, bias):
        """Returns (muscle activations (nu,), reserve torques (nv,) in Nm). Needs self.R at the current state."""
        m, d, nu, nv = self.m, self.d, self.m.nu, self.m.nv
        scale = self._torque_scale()
        A_tau = np.hstack([(self.R.T * gain) / scale[:, None], np.eye(nv)])
        y_tau = (tau - self.R.T @ bias) / scale
        w = np.sqrt(self.cfg.lam) * self.cost
        ws = np.sqrt(self.cfg.smooth)
        A_reg = np.zeros((2 * nu + nv, nu + nv))
        A_reg[:nu, :nu] = np.diag(w)
        A_reg[nu:2 * nu, :nu] = np.eye(nu) * ws
        A_reg[2 * nu:, nu:] = np.diag(self.reserve_w)
        A = np.vstack([A_tau, A_reg])
        y = np.concatenate([y_tau, w * cocon, ws * self.cfg.smooth_leak * self.a_prev, np.zeros(nv)])
        rmax = self.reserve_max / scale
        res = lsq_linear(A, y, bounds=(np.r_[np.zeros(nu), -rmax], np.r_[np.ones(nu), rmax]), method="bvls")
        self.last_residual = float(np.linalg.norm(A_tau @ res.x - y_tau) / np.sqrt(nv))
        self.a_prev = res.x[:nu]
        return res.x[:nu], res.x[nu:] * scale

    def excitation_for(self, a_des):
        """Invert MuJoCo's first-order muscle activation dynamics over one control tick."""
        act = self.d.act
        up = a_des > act
        tau = np.where(up, self.tau_act * (0.5 + 1.5 * act), self.tau_deact / (0.5 + 1.5 * act))
        k = 1 - np.exp(-self.cfg.horizon * self.dt / tau)
        return np.clip(act + (a_des - act) / k, 0.0, 1.0)

    def control(self, q_ref, qd_ref, qdd_ref, cocon, qfrc_load, support=None):
        """Compute (excitation, desired activation, reserve torque) for the next tick and apply the
        external load + support + reserve torque to qfrc_applied.

        qfrc_load: external joint torques (Nm), e.g. grip load.
        support:   (nv,) fraction of the *static* joint torque (gravity, passive muscle forces, joint springs) carried by an
                   external support such as an armrest; movements still have to be produced by the muscles.
        """
        m, d = self.m, self.d
        d.qacc[:] = qdd_ref + self.kp * (q_ref - d.qpos) + self.kd * (qd_ref - d.qvel)
        mujoco.mj_inverse(m, d)   # also refreshes moments, muscle lengths/velocities and M at the current state
        dense_moments(m, d, self.R)
        gain, bias = muscle_gain_bias(m, d)
        load = qfrc_load.copy()
        if support is not None:
            load += support * (d.qfrc_bias - self.R.T @ bias - d.qfrc_spring)
        tau = d.qfrc_inverse - load
        a_des, reserve = self.solve_activation(tau, cocon, gain, bias)
        d.qfrc_applied[:] = load + reserve
        return self.excitation_for(a_des), a_des, reserve
