# Validation

One script per stage of the pipeline. Each one checks that stage against something from outside myogen
(published measurements, physics, or open data), never against the datasets myogen is tuned on. That way a
stage that passes can be relied on, and a failure points to one stage. Reference values and tolerances are in
`references.yaml`, with their sources.

Run from this folder with the project's environment:
`for s in 0*.py; do ../../../.venv/bin/python $s; done`

| Script | Stage | Compared against |
|---|---|---|
| `01_muscle_actions.py` | musculoskeletal model | Textbook action of every muscle (sign and size of its moment arm). Joint limits vs. the functional range of motion (Ryu 1991, Morrey 1981). |
| `02_strength.py` | musculoskeletal model | Maximal isometric wrist moments (Delp et al. 1996). |
| `03_activations.py` | muscle solver (least squares) | Isometric loads: agonists answer, reserve torques stay small, the posture holds. Preferred directions fall in their anatomical quadrant. A relaxed hand stays silent. |
| `04_motor_units.py` | recruitment and rate coding | Firing rates, ISI variability, recruitment range (Fuglevand 1993, De Luca 2010). RMS vs. force exponent (Lawrence & De Luca 1983). Kurtosis vs. force (Nazarpour 2013). Spectrum. |
| `05_muap_physics.py` | MUAP to electrode | Biphasic engine vs. the layered volume conductor. `tests/test_volume_conductor.py` checks the volume conductor against the analytic half-space. |
| `06_hdemg_spatial.py` | MUAP to electrode, end to end | Correlation vs. distance on an 8x8 grid (10 mm spacing), real Hyser HD-EMG (PhysioNet, downloaded once into `~/.cache/myogen/hyser`). |
| `07_passive_release.py` | passive finger (musculoskeletal) | A relaxed index finger let go springs back to rest within 0.2 s, slightly under-damped (Xu et al. 2011). |
| `08_finger_force.py` | finger muscles (musculoskeletal) | Maximal single-finger pressing force per finger (Shinohara et al. 2003): the largest palmar fingertip force the muscles can hold (linear program over activations, with a friction cone). |

Work up the ladder: a failure in an early stage makes the later stages' results uninterpretable.
