# Realism plan: free-movement sEMG good enough to train on

Goal: myogen EMG that a pose decoder trained on it transfers to real recordings, for free movement (no load), fast
enough to generate hours of data. Every step is measured against real free-movement sessions (EMG-FK, NinaPro
DB2/DB7 exercise B, DB8) with the dataset repo's `scripts/tests/compare_synthetic.py`, replaying their recorded
joints at their own electrode positions.

## Where we are (decay 12 mm, average subject, Nov 2026)

| | real | myogen |
|---|---|---|
| per-movement ring pattern r (vs real-real between subjects) | 0.51-0.83 | 0.25-0.62 |
| ring RMS over time, r | - | 0.33-0.54 |
| neighbour channel r | 0.38-0.69 | 0.82-0.87 |
| rest / move RMS | 0.35-0.46 | 0.29-0.70 |
| autocorr 200 ms | 0.46-0.81 | 0.37-0.75 |

Causes, from the experiments:

- **Spatial smoothness**: each muscle is 6 point sources of Gaussian noise with an exponential gain. With an
  exponential the bipolar difference has the same spatial profile as a single pole, so bipolar and monopolar see
  equally far. A smaller decay lowers neighbour r but trades pattern r between datasets: not a parameter problem.
- **Texture and rest**: amplitude-modulated noise. Free movement is low force, where real sEMG is a few motor units
  firing: sparse, non-Gaussian, different under every electrode.
- **Holds and timing**: the drive is effort-minimal inverse dynamics low-passed at 4 Hz, plus random co-contraction.
- BioMime (motor-unit level, learned MUAPs) matched the correlation across the band on DB5 but was worse elsewhere
  and much slower: dropped. Speed is a constraint: today a 10 s motion takes 5.2 s (2.6 s of it the EMG step).

## Results so far (3 subjects per dataset; R²: joint decoder, trained on one repetition, tested on the other)

| | EMG-FK | DB2 | DB7 | DB8 |
|---|---|---|---|---|
| decoder real -> real | 0.19 | 0.36 | 0.32 | 0.45 |
| decoder other subject -> real (the yardstick) | -0.19 | -0.20 | -0.11 | -0.23 |
| decoder myogen -> real: noise, decay 12 | -0.57 | -0.19 | -0.76 | -2.78 |
| ... motor units, decay 12 | -0.36 | -0.24 | -0.54 | -3.38 |
| ... motor units, decay 8 | -0.29 | **-0.08** | -0.23 | -2.52 |
| ... motor units, decay 5 | -0.61 | **0.03** | -0.21 | -1.51 |
| ring pattern r, motor units decay 8 (real-real) | 0.70 (0.83) | 0.46 (0.51) | 0.21 (0.59) | 0.46 (0.56) |

- Motor units fixed the texture: median frequency 77-82 Hz -> 45-55 Hz (real 21-63), kurtosis 7-9 -> 10-14
  (real 7-27); bipolar is now more selective than monopolar. Same run time as the noise engine.
- With motor units the spread can shrink: decay 8 mm matches the real correlation vs distance on DB2 and transfers
  better than another real subject there. Default for the dataset check: motor units, decay 8.
- DB8 fails everywhere: its simulated rest is silent (rest / move 0.07 vs real 0.46) while real subjects keep a
  tone. That is step 4 (drive), now the largest gap.

### Known placement (calibration baseline frame, no search) and per channel

Turning myogen's ring to the best of 16 placements flattered it. With each real band placed by the calibration
baseline (found from the real data alone), pattern r myogen-real is 0.12-0.28 against 0.32-0.51 real-real
(NinaPro). Per channel, the ulnar side (225-270 deg) is nearly silent in myogen and strong in the recordings, the
dorsal one too strong, and the radial channel (90 deg) does not follow the movement (r 0.03). The electrodes placed
on FDS and EDC by palpation (DB2 / DB7, no placement involved) follow at r 0.29-0.45.

Cause: the muscle bellies are the proximal part of each muscle's line of action, not its volume. At the band
(proximal third), brachioradialis sits at 179 deg (dorsal; anatomically radial, ~90), ECRL at 189 (~120), FCU at
355 (~305), and nothing covers the ulnar sector but ECU. The forearm frame itself is right (radial styloid at 93 deg,
FCU / ECU insertions at 300 / 281). This makes step 5 (cross-sectional anatomy) the next one.

Tone (step 4): a flat rest tone of 0.04 per muscle (default) brought transfer to or above the other-subject
yardstick on EMG-FK, DB2 and DB7 (best-placement runs); a tone carried over from recent use and stronger
co-contraction did not help. The simulated rest / move ratio stays high (0.8 vs 0.4): the movement drive is low
(effort-minimal control engages few muscles), and MyoHand's fingers cannot hyperextend (MCP 0-1.57 rad), so
extension movements are clipped.

### Anatomy from an atlas (step 5, first part)

The muscles now sit where BodyParts3D has them (`EMGParams.anatomy = "atlas"`, `docs/anatomy.md`): each motor unit
at a point of its muscle's volume, moved by the model's belly with the posture. Under the band, BRD is radial-palmar
(-33 deg, was +67), ECRL radial (19, was 87), FCU ulnar (167, was -100), and the superficial muscles 6-9 mm below
the fat (were 12-21 mm). Placing the points is vectorised, with `belly_points`: 1.6 s for 10 s of 16-channel
motor-unit EMG (3.0 s before).

Known placement, 3 subjects, decoder R² myogen -> real (other subject -> real):

| | EMG-FK | DB2 | DB7 | DB8 |
|---|---|---|---|---|
| lines, decay 8 | -0.30 | 0.01 | -0.44 | -0.57 |
| **atlas, decay 8** | **-0.25** | **0.08** | **-0.06** | -0.47 |
| atlas, decay 12 | -0.18 | 0.08 | -0.17 | -0.74 |
| atlas, decay 16 | -0.17 | 0.04 | -0.23 | -0.86 |
| other subject (yardstick) | -0.75 | -0.20 | -0.33 | -0.38 |

- With the atlas, myogen transfers better than another real subject on EMG-FK, DB2 and DB7; DB8 is close.
- The electrodes over FDS by palpation (DB2 / DB7) follow better (r 0.29-0.30 -> 0.32-0.38); EDC stays at 0.36-0.43.
- Spatial spread: with muscles this superficial, decay 8 mm is too selective (r between electrodes 4 apart 0.19
  vs 0.54 real on DB2; the ring profile peaks 4x over ECRL / ECRB / EDC). Decay 16 matches the spread (0.35) and
  the ring pattern (DB8 0.21 vs 0.02) but transfers worse. The gain law (exponential, one decay for every depth) is
  the limit now: step 3 (layered cylinder, precomputed MUAP table) gives depth-dependent spread and shape.
- Frame: the calibration baseline anchors "wrist flexion peaks palmar"; in the anatomical atlas the wrist flexors'
  centre is ulnar-palmar (-169 deg). Turning myogen's ring by that same rule (`sim_anchor` in the dataset check)
  transfers worse (-0.89, -0.21, -0.52, -0.41): the anatomical frame is kept.
- Still open: the ulnar side is weaker in myogen than in the recordings, and rest / move stays high (0.9 vs 0.4).

### Why the atlas lowered the ring-pattern match, volume conductor, posture (Nov 2026)

Ring-pattern r myogen-real fell with the atlas on NinaPro (DB2 0.20 -> 0.00, DB8 0.28 -> 0.02) while transfer
rose. Turning myogen's ring by every step against the real bands (`compare_synthetic.py --rotations`) explains it:

- **Frame offset**: with the atlas, pattern r peaks one electrode (45°) away from 0 (DB2 0.24, DB7 0.33, DB8 0.20
  there; DB8 0.33 at 90°); the lines peaked at 0. The calibration baseline calls palmar where wrist flexion
  peaks; anatomically the flexor mass is ulnar-palmar. The lines were wrong in the way that matched that
  convention. Not an anatomy error: a frame definition the real bands and myogen do not share yet.
- **Posture**: no dataset records forearm rotation and myogen replayed every motion supinated (the atlas's pose),
  while NinaPro subjects rest the forearm on a desk. `Subject.pronation_rad` 1.5 (palm down) raises transfer on
  DB7 / DB8 (below).

Decoder R² myogen -> real (known placement, 3 subjects):

| | EMG-FK | DB2 | DB7 | DB8 |
|---|---|---|---|---|
| other subject (yardstick) | -0.75 | -0.20 | -0.33 | -0.38 |
| lines, biphasic | -0.30 | 0.01 | -0.44 | -0.57 |
| atlas, biphasic | -0.24 | 0.10 | -0.11 | -0.49 |
| **atlas, biphasic, pronation 1.5** | -0.29 | 0.06 | **-0.01** | **-0.19** |
| atlas, volume conductor | -0.56 | 0.07 | -0.10 | -0.70 |
| atlas, volume conductor, pronation 1.5 | -0.31 | 0.06 | -0.15 | -0.36 |

- Atlas + pronation beats another real subject on every dataset: the dataset check's setting now.
- The volume conductor (step 3, `docs/volume_conductor.md`) is the more realistic signal: ring-pattern r at 0°
  0.54 / 0.08 / 0.34 / 0.27 (biphasic 0.35 / -0.03 / 0.13 / 0.03), correlation 4 electrodes apart 0.32 on DB2
  (real 0.54, biphasic 0.18), rest / move on DB8 0.51 (real 0.46), the EDC electrode r 0.47. It transfers worse on
  EMG-FK (monopolar: the cylinder's far field reaches every pole) and DB8; kept as an option
  (`MotorUnitParams.waveform`) until that is understood. Its median frequency is 72 Hz after the datasets'
  filters (real 61-63 on DB2 / DB8); `iap_lambda` barely moves it (the end-plate / tendon edges dominate).
- Next: a shared frame (anchor myogen and the real bands the same way, or fit one rotation on NinaPro and check
  it on the others); the transfer experiment with `Subject.random` (pronation, section rotation, muscle
  variation all drawn) as augmentation, which is how the data would be used.

## Steps

### 1. Validation metrics (dataset repo + myogen statistics)

Add to the comparison, real vs simulated:

- correlation vs ring distance (1, 2, 3, 4 electrodes apart), instead of neighbours only;
- kurtosis of the raw EMG (Gaussian 3; motor-unit trains higher at low force);
- median frequency;
- EMG -> movement lag (envelope vs joint speed cross-correlation peak);
- **transfer**: a ridge decoder (log RMS -> joints in 0-1) trained on myogen, tested on the real EMG of the same
  movements, against one trained and tested on real (first half / second half). This is the number that says
  whether the data helps training.

### 2. Motor-unit pools (myogen `simulation/motor_units.py`, replaces the noise sources)

Per muscle (extrinsic forearm muscles; the intrinsic hand muscles are outside the forearm and stay silent):

- **Pool**: N units (default ~120, scaled by muscle volume), recruitment thresholds exponential over 0-0.7 drive
  (Fuglevand 1993), larger units recruited later, with larger territories and amplitudes.
- **Firing**: rate 8 Hz at recruitment rising 30-50 Hz/unit-drive to 35-50 Hz; inter-spike intervals Gaussian,
  CV 0.15; a common drive fluctuation shared by the units of a muscle (1-5 Hz, small) gives realistic coherence.
  Spikes come straight from the drive at the control rate: no ODEs.
- **Territory**: each unit at a fixed position inside its muscle cross-section, around the belly path (as the source
  offsets today), so unit positions move with the posture like the bellies do.
- **MUAP at a pole**: amplitude from the distance (current gain law at first), shape a propagating-potential
  waveform widened with distance (tissue low-pass: width ~ distance / conduction velocity). Far units are wide and
  slow, so the bipolar difference cancels them: bipolar is more selective than monopolar, as in reality.

**Fast by construction**: spikes are sparse. Each spike adds its gain into one of a few width bands at every pole
(posture-dependent gains per control tick); each band is then convolved once with its waveform. Cost ~ spikes x
poles x bands, about the current EMG step. Deterministic per subject seed.

Check after step 2: neighbour r and correlation vs distance, kurtosis, rest / move, pattern r, transfer, run time.

### 3. Volume conductor (if step 2 plateaus)

Analytic multilayer cylinder (bone, muscle, fat, skin; anisotropic muscle): MUAP templates per unit and electrode,
including generation and extinction at the end plate and tendon, computed per subject at a few postures and
interpolated. Replaces the gain + width approximation of step 2 behind the same interface.

### 4. Neural drive

Co-contraction from the joint stiffness the movement needs and its speed (instead of random episodes); postural tone
in holds; signal-dependent drive noise per muscle; electromechanical delay. Fitted to the real rest / hold / move
ratios and lags of the free datasets: unloaded holds need no force data.

### 5. Anatomy and devices

- Muscle cross-sections (done: BodyParts3D atlas, above; next: per-subject variation). Before, the bellies were the tendon path lines of the MSK model, scaled into a cylinder. Fit
  each muscle's position and area at several levels along the forearm to a cross-sectional atlas (MRI / Visible
  Human): what sits under each electrode is what the pattern r measures.
- Subject priors (forearm size, fat, muscle sizes) from anthropometric ranges; band placement (rotation, along-arm
  shift, tilt); per-channel contact; dry-electrode noise and slip artifacts; the target band's profile from a short
  recording (`calibrate`).

### 6. Motion

Pronation / supination and arm posture (gravity loads change the EMG); later a motion generator trained on all
free-movement kinematics to sample new movements instead of replaying.

## Not changing

MuJoCo and the MyoSim forearm stay. The mechanics are not where the EMG gap is: the model already has every
extrinsic forearm muscle with MoBL-ARMS paths (MyoSim's origin), MuJoCo is the fast option, and an OpenSim model would
give the same muscle lines more slowly. What would help is the anatomy *around* the lines (step 5), not another
engine.
