# Pipeline: from hand kinematics to muscle drive to EMG

`Simulator._clean_one` (simulator.py) runs it; every number below is a default of `SimParams` (params.py) or
`Subject` (inputs/subject.py). The scope it works in: [scope.md](scope.md).

```
hand angles (23 joints, any rate)                                     inputs/motion.py
  │ 1  joint trajectory: 100 Hz, clipped to the model's range, warm-up, elbow and forearm, relaxed posture
q_ref (T, nq), q_rest                                                 Simulator.joint_trajectory
  │ 2  loads: gravity, armrest, passive joint springs at the relaxed posture
  │ 3  tracking controller, every 10 ms:
  │      desired acceleration ─► joint torques ─► muscle activations ─► excitations ─► MuJoCo step
excitation (T, 48)                                                    MuscleController.track
  │ 4  neural drive = excitation low-passed at 4 Hz
  │ 5  + effort beyond the mechanical minimum (co-activation)          effort
  │ 6  + rest tone                                                    with_tone
drive (T, 48) in [0, 1], qpos (T, nq): MuscleActivity                 simulation/muscle_activity.py
  │ 7  motor units: recruitment and firing from the drive
  │ 8  each unit's action potential at every pole, from where it sits in the muscle's cross-section
  │ 9  derivation (bipolar, or monopolar against the ring average), device
EMG (T·20, channels) at 2 kHz                                         simulation/emg.py, motor_units.py, acquisition.py
```

## Kinematics -> muscle drive

1. **Joint trajectory** (`Simulator.joint_trajectory`). The 23 hand angles (rad) are resampled to the control rate
   (100 Hz) and clipped to the model's joint range (warning beyond 0.05 rad). Joints not given are filled
   (`JointAdapter`: a finger's MCP and PIP from each other, its DIP from the PIP, anything else at `REST_POSTURE`).
   The first posture is held 0.5 s (warm-up, dropped at the end). The elbow sits at `Subject.elbow_angle_deg` (72°);
   the forearm rotation (`pro_sup_r`) follows the motion if tracked, else stays neutral. The relaxed posture
   `q_rest` is the median posture of the motion's rest frames (when it has at least 1 s of them), else
   `REST_POSTURE`.
2. **Loads** (`ForearmModel.prepare`). Gravity points down, perpendicular to the forearm (forearm on an armrest).
   The armrest carries most of the static torque (elbow 92.5 %, wrist 60 %). Every joint has a passive spring whose
   equilibrium is the relaxed posture (capsules, ligaments, resting tone), offset so that gravity and the muscles'
   passive forces are balanced there: **a relaxed hand needs no drive**. No grip torque (free movement).
3. **Tracking controller** (`MuscleController`), per 10 ms tick:
   - desired acceleration = the reference's acceleration + PD feedback on its angle and speed (8 Hz bandwidth,
     critically damped);
   - joint torques for it: MuJoCo inverse dynamics (`mj_inverse`), which also gives the muscles' moment arms `R`
     and their force-length-velocity state (`force = gain · activation + bias`);
   - **muscle activations** (48, in [0, 1]) by bounded least squares (static optimisation):
     `min ‖Rᵀ(gain·a + bias) + reserve − τ‖² + 3e-3 ‖a‖² + ‖a − 0.9 a_prev‖² + ‖reserve‖²`,
     torque errors weighted as acceleration errors (light finger joints tracked as tightly as the wrist). The
     effort term shares the work between muscles, the second keeps the activations smooth (unneeded drive decays in
     ~100 ms). Reserve torques (≤ 0.5 Nm) take what no muscle can do; the thumb, whose intrinsic muscles the model
     lacks, gets cheap, larger ones (≤ 3 Nm);
   - excitations that reach those activations: MuJoCo's first-order activation dynamics inverted over 3 ticks;
   - MuJoCo steps forward with them, so the motion is physically consistent (reset to the reference if it diverges).
4. **Neural drive**: the excitations low-passed at 4 Hz (2nd order, zero phase): the controller's fast corrections
   are a simulation artifact, real drive changes slower.
5. **Effort** (`effort`): people drive their muscles beyond the mechanical minimum. Each joint is *engaged* while it
   moves (speed beyond 0.15 range/s) or is held away from the relaxed posture (beyond 0.1 of its range), signed by
   direction. Each muscle gets the engagement of the joints it acts on, weighted by its moment arm relative to the
   joint's strongest muscle: in full when it pulls that way (agonist), 0.25 when it pulls the other way
   (antagonist). Level: hold tone 0.15 + co-contraction 0.02 with random episodes (up to 0.3, 0.5-3 s), plus 0.125
   x the joint's speed, except back towards the relaxed posture (the hand is let go there, not driven). Added to
   the drive only, not to the mechanics.
6. **Rest tone**: every muscle keeps 0.02 of drive (`Subject.rest_tone`). Below the first recruitment threshold
   (0.7 / 30 ≈ 0.023): no motor unit fires on tone alone.

The result, `MuscleActivity`, holds the drive of the 48 muscles per tick (and the joint state the EMG step needs for
the muscles' positions).

## Muscle drive -> EMG

7. **Motor units** (`MotorUnits`, extrinsic muscles only: the hand's own muscles are not under a forearm band): 0.25
   units per N of the muscle's maximal force (20-200). Recruitment thresholds spread exponentially up to 0.7 of the
   drive (Fuglevand 1993), firing 8 Hz at recruitment rising to 35 Hz, inter-spike CV 0.15, a 5 % common drive.
8. **At the electrodes**: each unit sits at a point of its muscle's cross-section (atlas anatomy,
   [anatomy.md](anatomy.md)), moved with the joint state. Its action potential at a pole: an amplitude falling with
   distance (e^(−d / 8 mm)) and a biphasic waveform widening with it (or, as an option, the layered volume
   conductor, [volume_conductor.md](volume_conductor.md)). Each muscle's units are scaled so that the muscle fully
   driven under a pole gives an RMS of `Subject.muscle_amp` x (its strength factor)^0.5: by default the same for
   every muscle, whatever its size or how much of it reaches the skin.
9. **Derivation and device**: bipolar pairs (10 mm), or monopolar poles against the ring average; then the
   dataset's device (`acquisition.record`: amplifier band-pass, noise, mains, movement artifacts).

The EMG settings above are those of the checks (`synthetic.yaml` in the dataset repo: motor-unit engine, atlas
anatomy, decay 8 mm).
