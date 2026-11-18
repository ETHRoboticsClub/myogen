# Scope: what myogen is for, and what is fixed

myogen turns recorded hand kinematics into the forearm surface EMG that produced them. The points below are locked:
work on myogen improves it within them, and data outside them is neither a target nor an input.

## Locked

1. **Free movement only.** The hand moves in the air: no object, no contact, no external load. Forces are not
   modelled (no force data to drive or check them): `Subject.grip_load` is 0.
2. **The kinematics must include the wrist and the lateral movements**, so the muscle activations can be inferred:
   wrist flexion / extension and radial / ulnar deviation, the finger flexions (MCP, PIP; DIP optional), finger
   abduction / adduction, and the thumb (CMC flexion and abduction, MCP, IP). A recording without them is not used.
3. **Forearm rotation only when tracked** (`pro_sup_r`); otherwise it is held neutral, and movements that rotate the
   forearm are left out.
4. **Arm posture fixed**: elbow at `Subject.elbow_angle_deg` (72°), forearm on a support, gravity down. The upper arm
   does not move; the elbow muscles only hold the posture.
5. **One reference subject for every check** (`Subject()`: strength 1, fat 5 mm, right arm, intact). Subject
   variation (`Subject.random`) is for augmenting training data, never for validation.
6. **One ring of 8 electrodes**, placed by the calibration baseline's frame and `ring_offset_deg` (-45°). The
   placement is not tuned per dataset or per subject.
7. **What has to match: the EMG envelope per channel** (RMS in 100 ms bins): its level around the ring and its course
   over time. Raw waveforms, spectra and action-potential shapes only have to be plausible.
8. **The musculoskeletal model and the controller are frozen** (MyoSim forearm, 48 muscles, its moment arms;
   [pipeline.md](pipeline.md)). Iteration 5 ([iterations.md](iterations.md)) showed the match does not depend on
   their settings, and the muscle drives they give carry as much of the recorded EMG as the kinematics do
   (iteration 8). What is fitted is the step from drives to electrodes, and every fit is checked on held-out subjects.

## Targets and scoring

- **Data**: the free-movement sessions that meet 1-3 and have a known band placement: NinaPro DB2 and DB7 exercise B
  (without its forearm rotations, cues 9-12) and DB8 (A1-A3, movements mimed without objects). EMG-FK has no wrist
  nor abduction tracking (camera, finger flexions only): not a target. Dataset repo,
  `src/femg/config/synthetic.yaml` (`required_joints`, `excluded_cues`).
- **Split**: subjects drawn at random per dataset, a tune set for every decision and a validate set never tuned on.
- **References**: a channel is scored against what the kinematics can explain at all: a regression kinematics ->
  EMG fitted on the same subject (the ceiling) and on the other subjects (another person: the target for myogen).

## Accepted limitations

- The hand's own (intrinsic) muscles give no EMG at a forearm band; the thumb's are replaced by reserve torques.
- Joint angles beyond the model's range are clipped (MCP hyperextension).
- No fatigue, no electrode shift with movement, no change of skin impedance over a session.
