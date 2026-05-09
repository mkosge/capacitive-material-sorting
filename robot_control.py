"""
Full pick → measure → classify → sort pipeline.

Flow:
  1. Connect to arm
  2. Move to observation pose, detect object
  3. Top-down pick and grip the capacitor
  4. Move to measurement pose, holding the cap
  5. PAUSE — user takes AD7746 capacitance measurement, saves data file
  6. Script reads the file, runs material_classifier_3 → Metal / Glass / Paper
  7. Print prediction, user confirms or corrects
  8. Drop the cap in the bin matching the (confirmed) material
  9. Return arm to horizontal-flat reset pose

Requires classifier.py in the same folder.
The AD7746 measurement / data export is performed by you separately;
this script only reads the resulting file.
"""

import os
import sys
import time

from pyniryo import NiryoRobot, PoseObject, ObjectShape, ObjectColor

# Ensure classifier.py is importable regardless of where the script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ensures classifier.py is found
from classifier import parse_file, extract_features, classify


# ── Robot connection ────────────────────────────────────────────────────────
ROBOT_IP = "10.10.10.10"
WORKSPACE_NAME = "testing"

# ── Pick height (tune in 5 mm steps) ────────────────────────────────────────
PICK_HEIGHT = 0.035

# ── Cap-vs-detection XY offset ──────────────────────────────────────────────
# Vision detects the HOLDER's centre. If the camera or workspace calibration
# is off by a few mm, the gripper will descend slightly off the cap. Run
# snap_pre_pick_view.py to see what the camera sees at the pre-pick height,
# measure how far the cap is from the image centre, and put the correction
# in metres here. Positive X = further from robot base; positive Y = robot's
# left side. Start at 0 and tune.
CAP_OFFSET_X = 0.000
CAP_OFFSET_Y = 0.000

# ── Detection / motion ──────────────────────────────────────────────────────
DETECTION_SAMPLES = 3
LIFT_BEFORE_PICK = 0.05
LIFT_AFTER_PICK  = 0.06

DEBUG_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_debug.jpg")

# ── Poses ───────────────────────────────────────────────────────────────────
# Observation pose. Tune z so all four workspace corner markers are visible
# (use Niryo Studio's Vision panel as ground truth).
OBSERVATION_POSE = PoseObject(
    x=0.15, y=0.00, z=0.18,
    roll=0.0, pitch=1.57, yaw=0.0,
)

# Where the arm holds the cap during the AD7746 measurement.
# *** TUNE this to wherever your sensor station is on the bench. ***
MEASUREMENT_POSE = PoseObject(
    x=0.20, y=0.15, z=0.20,
    roll=0.0, pitch=1.57, yaw=0.0,
)

# Three drop locations — one per material. Tune to your bench.
# Default layout: a row in front of the robot, METAL on the +Y side, PAPER on -Y.
METAL_DROP_POSE = PoseObject(
    x=0.10, y=0.18, z=0.18,
    roll=0.0, pitch=1.57, yaw=0.0,
)
GLASS_DROP_POSE = PoseObject(
    x=0.10, y=0.00, z=0.18,
    roll=0.0, pitch=1.57, yaw=0.0,
)
PAPER_DROP_POSE = PoseObject(
    x=0.10, y=-0.18, z=0.18,
    roll=0.0, pitch=1.57, yaw=0.0,
)

DROP_POSES = {
    "Metal": METAL_DROP_POSE,
    "Glass": GLASS_DROP_POSE,
    "Paper": PAPER_DROP_POSE,
}

# Reset pose: same XY as observation, lower z. Arm settles here after the
# routine completes (after the brief return to observation). Tune z to taste.
RESET_POSE = PoseObject(
    x=OBSERVATION_POSE.x, y=OBSERVATION_POSE.y, z=0.10,
    roll=0.0, pitch=1.57, yaw=0.0,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def save_camera_view(robot, filename):
    """Save a JPEG of what the camera currently sees, for debugging."""
    try:
        img = robot.get_img_compressed()
        with open(filename, "wb") as f:
            f.write(img)
        print(f"  Saved camera view to: {filename}")
    except Exception as e:
        print(f"  Could not save camera image: {e}")


def average_detections(robot, samples):
    """Take N detections of the workspace object and average them.

    Wraps each call in try/except KeyError to swallow the pyniryo bug where
    an empty-string enum lookup crashes when nothing is detected.
    """
    xs, ys, yaws = [], [], []
    last_shape, last_color = None, None
    for i in range(samples):
        try:
            obj_found, rel_pose, shape, color = robot.detect_object(
                WORKSPACE_NAME,
                shape=ObjectShape.ANY,
                color=ObjectColor.ANY,
            )
        except KeyError:
            print(f"  sample {i + 1}: nothing in frame (pyniryo enum bug)")
            time.sleep(0.2)
            continue
        if obj_found:
            x_rel, y_rel, yaw_rel = rel_pose
            xs.append(x_rel)
            ys.append(y_rel)
            yaws.append(yaw_rel)
            last_shape, last_color = shape, color
            print(
                f"  sample {i + 1}: "
                f"x={x_rel:.3f}, y={y_rel:.3f}, yaw={yaw_rel:.3f}"
            )
        else:
            print(f"  sample {i + 1}: not found")
        time.sleep(0.1)
    if not xs:
        return None
    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(yaws) / len(yaws),
        last_shape,
        last_color,
    )


def prompt_for_classification():
    """Pause for the user to take a measurement and save a data file.

    Reads the file, runs the classifier, prints the prediction, and asks
    the user to confirm or override it. Returns one of "Metal" / "Glass" /
    "Paper", or None if the user skips.
    """
    print()
    print("=" * 60)
    print("MEASUREMENT TIME — arm is holding the capacitor.")
    print("Take your AD7746 reading now and save the data to a text file.")
    print("=" * 60)

    # ── Get a valid file path with a successful classification ──────────────
    while True:
        path = input(
            "\nPath to data file or its folder (or 'skip' to bypass): "
        ).strip()
        if path.lower() == "skip":
            return None
        # Strip quotes if the user pasted a Windows path with quotes
        path = path.strip('"').strip("'")

        # Be tolerant of common mistakes:
        #   - user gave a path without .txt extension
        #   - user gave the folder, not the file → auto-pick most recent .txt
        if not os.path.exists(path) and os.path.isfile(path + ".txt"):
            path = path + ".txt"
            print(f"  (using {os.path.basename(path)})")
        elif os.path.isdir(path):
            txt_files = [
                f for f in os.listdir(path) if f.lower().endswith(".txt")
            ]
            if not txt_files:
                print(f"  No .txt files in: {path}")
                continue
            # Use the most recently modified .txt — handy when you keep
            # exporting fresh AD7746 readings into the same folder.
            txt_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(path, f)),
                reverse=True,
            )
            path = os.path.join(path, txt_files[0])
            print(f"  Auto-selected most recent: {os.path.basename(path)}")

        if not os.path.isfile(path):
            print(
                f"  File not found: {path}\n"
                f"  Tip: include the file extension (e.g. '.txt'), "
                f"or paste the folder path and I'll grab the most recent .txt."
            )
            continue

        try:
            readings, raw_ints = parse_file(path)
            if not readings:
                print("  No valid readings parsed from that file.")
                continue
            avg, noise, sat_frac = extract_features(readings, raw_ints)
            prediction = classify(avg, noise, sat_frac)
            print(f"\n  Average:          {avg:.4f} pF")
            print(f"  RMS noise:        {noise:.4f} pF")
            print(f"  Saturation frac.: {sat_frac:.1%}")
            print(f"  Prediction:       {prediction}")

            # Stage 1 already handles saturation in classify(), but warn
            # the operator so they can verify the sensor connection.
            if sat_frac > 0.50:
                print(
                    "  WARNING: >50% of samples are at the ADC rail. "
                    "If this is not a metal object, check the sensor connection."
                )
            break
        except Exception as e:
            print(f"  Error parsing file: {e}")
            continue

    # ── Confirm or correct the prediction ───────────────────────────────────
    valid = ("Metal", "Glass", "Paper")
    while True:
        response = input(
            f"\nIs '{prediction}' correct? "
            f"(y / n / Metal / Glass / Paper): "
        ).strip()
        lower = response.lower()
        if lower in ("y", "yes", ""):
            return prediction
        if lower in ("n", "no"):
            # Fall through and ask which material it actually is
            response = input("  Correct material (Metal / Glass / Paper): ").strip()
            lower = response.lower()
        if lower == "metal":
            return "Metal"
        if lower == "glass":
            return "Glass"
        if lower == "paper":
            return "Paper"
        print(f"  Please type one of: y, n, {', '.join(valid)}")


# ── Main routine ────────────────────────────────────────────────────────────

def main():
    # 1. Connect
    print(f"Connecting to {ROBOT_IP}...")
    robot = NiryoRobot(ROBOT_IP)

    try:
        robot.set_learning_mode(False)
        robot.calibrate_auto()
        robot.clear_collision_detected()
        robot.update_tool()
        robot.set_arm_max_velocity(20)   # match snap_camera_view.py

        # 2. Observation pose + open gripper + camera settle
        print("Moving to observation pose...")
        robot.move(OBSERVATION_POSE)
        robot.open_gripper()
        time.sleep(0.5)

        # 3. Detect
        print(f"Detecting object ({DETECTION_SAMPLES} samples averaged)...")
        result = average_detections(robot, DETECTION_SAMPLES)
        if result is None:
            print(
                "No object detected — aborting.\n"
                "  Check: holder centred, all four workspace corner markers\n"
                "         visible (open the saved camera_debug.jpg to see what\n"
                "         the camera saw)."
            )
            save_camera_view(robot, DEBUG_IMAGE_PATH)
            return
        x_rel, y_rel, yaw_rel, shape, color = result
        print(f"Averaged: shape={shape}, color={color}")

        # 4. Pick (top-down)
        target = robot.get_target_pose_from_rel(
            WORKSPACE_NAME, height_offset=PICK_HEIGHT,
            x_rel=x_rel, y_rel=y_rel, yaw_rel=yaw_rel,
        )
        # Apply XY offset for camera/workspace calibration error if any.
        grasp = PoseObject(
            x=target.x + CAP_OFFSET_X,
            y=target.y + CAP_OFFSET_Y,
            z=target.z,
            roll=target.roll, pitch=target.pitch, yaw=target.yaw,
        )
        above_pick = PoseObject(
            x=grasp.x, y=grasp.y, z=grasp.z + LIFT_BEFORE_PICK,
            roll=grasp.roll, pitch=grasp.pitch, yaw=grasp.yaw,
        )
        lift = PoseObject(
            x=grasp.x, y=grasp.y, z=grasp.z + LIFT_AFTER_PICK,
            roll=grasp.roll, pitch=grasp.pitch, yaw=grasp.yaw,
        )
        print(f"Top-down pick at z={grasp.z * 1000:.1f} mm...")
        robot.move(above_pick)
        robot.move(grasp)
        robot.close_gripper(speed=500)
        robot.move(lift)

        # 5. Move to measurement station, holding the cap
        print("Moving to measurement pose...")
        try:
            robot.move(MEASUREMENT_POSE)
        except Exception as e:
            print(f"Failed to reach measurement pose: {e}")
            return

        # 6+7. Pause for user, classify from data file, confirm/correct
        material = prompt_for_classification()
        if material is None:
            print("Classification skipped — defaulting to Metal drop.")
            material = "Metal"

        # 8. Drop based on material
        drop_pose = DROP_POSES[material]
        print(f"\nDropping at {material} location ({drop_pose.x:.2f}, "
              f"{drop_pose.y:.2f}, {drop_pose.z:.2f})...")
        try:
            robot.move(drop_pose)
        except Exception as e:
            print(f"Failed to reach drop pose: {e}")
            return
        robot.release_with_tool()

        # 9. Return to observation pose, ready for the next cycle
        print("Returning to observation pose...")
        try:
            robot.move(OBSERVATION_POSE)
        except Exception as e:
            print(f"Failed to return to observation pose: {e}")
            return

        # 10. Settle into reset pose (same XY as observation, lower z)
        print("Lowering into reset pose...")
        try:
            robot.move(RESET_POSE)
        except Exception as e:
            print(f"Failed to lower into reset pose: {e}")
            return

        print("Done.")

    finally:
        robot.set_learning_mode(True)
        robot.close_connection()


if __name__ == "__main__":
    main()
