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

- Muscle cross-sections: today the bellies are the tendon path lines of the MSK model, scaled into a cylinder. Fit
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
