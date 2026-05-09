# Capacitive Material Sorting with the Niryo Ned2

A robotic pick-and-sort system that classifies small objects by their dielectric response to a capacitive sensor and routes them to material-specific bins.

A six-degree-of-freedom Niryo Ned2 collaborative arm picks an object from a fixed workspace, holds it across the plates of an AD7746 24-bit capacitance-to-digital converter, and a Python classifier identifies the material (Metal, Glass, or Paper) from the measured capacitance. The arm then drops the object in the bin corresponding to the predicted material. The system was developed as a final-year individual project at the University of Manchester (EEEN30330) to investigate whether a low-cost, single-channel capacitive sensor is a viable front-end for material-aware robotic sorting in waste-stream and recycling applications.

The repository contains two Python scripts: `robot_control.py` (arm control and workflow orchestration) and `classifier.py` (data-file parsing, feature extraction, and two-stage classification). Capacitance acquisition is performed by the Analog Devices EVAL-AD7746EBZ evaluation-board GUI; this project does not include custom AD7746 acquisition firmware. Object detection uses the built-in Niryo Studio workspace vision system.

## System overview

<!-- INSERT PHOTO OR DIAGRAM HERE — recommended: a labelled photo of the rig
     showing the Niryo arm, the AD7746 evaluation board, the sensor plate
     fixture, the workspace tile, and the three bins.
     A short demo video linked here satisfies the supervisor's demo requirement. -->

The high-level flow is:

1. Operator places an object on the calibrated Niryo workspace.
2. `robot_control.py` connects to the arm, moves to the observation pose, runs three vision detections and averages the result.
3. The arm grasps the object top-down and lifts it to the measurement station.
4. The script pauses; the operator triggers a 100-sample acquisition in the AD7746 GUI and exports the result as a tab-delimited text file.
5. The script reads the file, runs the two-stage classifier, and prints the prediction.
6. The operator confirms or corrects the prediction via keyboard input.
7. The arm drops the object in the bin corresponding to the confirmed material and returns to the observation pose.

## Hardware required

- Niryo Ned2 collaborative robotic arm with two-finger electric gripper, on a level bench.
- Niryo Studio workspace tile with four positioning markers visible to the arm-mounted Vision Set camera.
- Object holder positioned on the workspace, sized to receive the test capacitor housings.
- Analog Devices EVAL-AD7746EBZ capacitance-to-digital evaluation board, connected to the host PC via USB.
- A fixed pair of sensor plates wired to the AD7746 CIN1+/CIN1− pins, mounted so the arm can lower the gripped object cleanly between the plates without contact. See Section 3.4 of the project report for the mechanical arrangement.
- Three sorting bins in a row in front of the robot. Default layout: Metal at y = +0.18 m, Glass at y = 0, Paper at y = −0.18 m (all at x = 0.10 m, z = 0.18 m in the robot base frame).
- Test objects: small capacitor housings of three nominal materials — Metal (aluminium body), Glass (glass-bodied capacitor), Paper (paper-foil capacitor) — sized identically so the gripper closes consistently.

## Software required

- Windows 10 or 11 (the AD7746 evaluation-board GUI is Windows-only).
- Python 3.9 or newer.
- Python packages: `pyniryo`, `numpy` (see Installation).
- [Niryo Studio](https://niryo.com/download/) for workspace calibration and camera diagnostics.
- [Analog Devices EVAL-AD7746EBZ evaluation software](https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/eval-ad7746.html) for capacitance acquisition and text-file export.

## Installation (from scratch)

These instructions assume a fresh Windows installation with neither Python nor any robot tooling present.

**1. Install Python.**

Download Python 3.11 (or newer) from [python.org/downloads](https://www.python.org/downloads/). During installation, tick **"Add Python to PATH"**. Verify:

```
python --version
```

You should see `Python 3.11.x` or similar.

**2. Clone this repository.**

```
git clone https://github.com/MatthewKosgei/capacitive-material-sorting.git
cd capacitive-material-sorting
```

**3. Create a virtual environment (recommended).**

```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your command prompt.

**4. Install Python dependencies.**

```
pip install -r requirements.txt
```

This installs `pyniryo` and `numpy`.

**5. Install Niryo Studio.**

Download from [niryo.com/download](https://niryo.com/download/) and follow the installer.

**6. Install the AD7746 evaluation-board software.**

Download from the Analog Devices product page linked above and follow the installer. Plug the EVAL-AD7746EBZ into the host PC via USB; the GUI detects it automatically.

**7. Calibrate the Niryo workspace.**

In Niryo Studio:
- Connect to the Ned2 (default hotspot IP `10.10.10.10`).
- Open the Vision panel.
- Place the workspace tile in front of the robot with all four corner markers clearly visible to the camera.
- Run the workspace calibration routine and save the workspace under the name **`testing`** (the name expected by `WORKSPACE_NAME` in `robot_control.py`; change this constant if you prefer a different name).

**8. Configure the AD7746 GUI.**

- Set the channel to **CIN1+/CIN1−** (single-ended).
- Set the conversion time to **109.6 ms** (16.1 Hz update rate, used in this project).
- Set excitation voltage to the default `+VDD/2`.
- Configure data export to write tab-delimited text with the 24-bit hex code in the first column.

**9. Tune pose constants.**

Open `robot_control.py` and adjust `OBSERVATION_POSE`, `MEASUREMENT_POSE`, `METAL_DROP_POSE`, `GLASS_DROP_POSE`, and `PAPER_DROP_POSE` to match your bench. `PICK_HEIGHT` (default 35 mm) sets gripper descent depth; tune in 5 mm increments starting just above the workspace surface.

## Running the system

With the Ned2 powered on, the AD7746 connected, and the GUI open and ready to capture:

```
python robot_control.py
```

The script:

1. Connects, calibrates, and moves to the observation pose.
2. Detects and grasps the object.
3. Moves to the measurement pose and pauses:
   ```
   Path to data file or its folder (or 'skip' to bypass):
   ```
4. At this prompt, switch to the AD7746 GUI, click **Start** (100 samples), click **Save** to export a `.txt` file, then paste the path (or just the folder — the script auto-picks the most recent `.txt`).
5. The classifier runs and prints the prediction with saturation fraction.
6. Confirm with `y` or override with `Metal`, `Glass`, or `Paper`.
7. The arm drops the object and returns to the observation pose.

**Classifier only (offline analysis):**

```
python classifier.py path/to/your_data.txt
```

**Self-test against calibration files:**

Place `paper_results.txt`, `glass_results.txt`, and `metal_results.txt` in a `data/` subfolder, then:

```
python classifier.py --selftest
```

Note: this evaluates against the same three files used to derive the thresholds — it is a training-set fit check, not a held-out validation result.

## Technical details

### Capacitance conversion — AD7746 hex to picofarads

The AD7746 encodes its 24-bit result with 0x800000 as zero capacitance and ±8.192 pF full-scale range:

$$C_\text{raw} \,[\text{pF}] = \left( \frac{N_\text{hex}}{2^{23}} - 1 \right) \times 8.192$$

### Linear calibration correction

The GUI applies undocumented internal corrections. A linear correction is fitted from two anchor measurements (Paper and Metal) taken directly from the GUI display:

$$C_\text{corrected} = a \cdot C_\text{raw} + b$$

$$a = \frac{C_\text{GUI,Metal} - C_\text{GUI,Paper}}{C_\text{raw,Metal} - C_\text{raw,Paper}}, \qquad b = C_\text{GUI,Paper} - a \cdot C_\text{raw,Paper}$$

Numeric values are defined as constants at the top of `classifier.py`. Glass is interpolated; it is not an anchor point.

### Two-stage classifier

**Stage 1 — Saturation test (metal detection).** A sample is considered saturated if its 24-bit integer value is within 500 counts of either ADC rail (0x000000 or 0xFFFFFF). If more than 80% of the 100-sample run is saturated, the object is classified as Metal immediately. This reflects the empirical finding that a metal-bodied capacitor shorts the sensor plates at the AD7746 excitation frequency, driving the converter to its rails. See Section 4.3 of the project report for the supporting measurement data.

**Stage 2 — Mean-capacitance thresholds (dielectric discrimination).** If the saturation rate is below 80%, the corrected mean capacitance $\bar{C}$ is compared against two midpoint thresholds:

$$\text{Material} = \begin{cases} \text{Glass} & \text{if } \bar{C} > T_\text{MG} \\ \text{Paper} & \text{otherwise} \end{cases}$$

with $T_\text{MG} = -1.675\,\text{pF}$ (Glass/Metal boundary) and $T_\text{GP} = -1.876\,\text{pF}$ (Glass/Paper boundary). Thresholds are midpoints of per-class calibration means; see Section 4.3.

### Pose model and pick geometry

`pyniryo.get_target_pose_from_rel` transforms workspace-relative detection coordinates to the robot base frame, with a configurable height offset (`PICK_HEIGHT = 35 mm`). The grasp pose is approached via a vertical pre-grasp 50 mm above (`LIFT_BEFORE_PICK`); after gripper close, the object is lifted 60 mm (`LIFT_AFTER_PICK`) before moving to the measurement station.

## Known issues and future improvements

**Single-cycle execution.** `main()` in `robot_control.py` runs one pick-place-sort cycle and exits. Wrap the body in `while True:` for batch operation.

**Skip defaults to Metal bin.** Typing `skip` at the classification prompt silently routes the object to the Metal bin — a leftover from early debugging. It should re-prompt or abort cleanly. (Issue #1.)

**Robot IP is hardcoded.** `ROBOT_IP = "10.10.10.10"` is the Ned2 default hotspot address. Change this constant if running over Ethernet.

**Pyniryo `detect_object` enum bug.** When no object is in frame, `detect_object` raises a `KeyError` from an empty-string enum lookup instead of returning `obj_found = False`. The `average_detections` helper wraps each call in `try/except KeyError` to handle this; if a future pyniryo release fixes the bug, the wrapper becomes redundant.

**No held-out validation.** Classifier thresholds are derived from the same three files used by `--selftest`. The reported 3/3 accuracy is a training-set fit, not a generalisation result. A meaningful evaluation requires fresh measurements across multiple sessions; see Section 4.5 of the report for discussion.

**Manual measurement step.** The system pauses for a human operator to trigger acquisition through the AD7746 GUI. Full automation would require driving the AD7746 directly over I²C or scripting the GUI — out of scope for this project.

## Authors and acknowledgements

Matthew Kosgei, BEng Electrical and Electronic Engineering, University of Manchester.  
Project supervisor: Professor Wuqiang Yang.  
Submitted as part of EEEN30330 Individual Project, 2025–2026.

## Licence

MIT — see `LICENSE` file.
