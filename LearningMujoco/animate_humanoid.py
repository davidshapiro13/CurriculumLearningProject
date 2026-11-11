from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def animate_joint_ranges(model, output_path="joint_ranges.mp4", fps=30,
                         seconds_per_joint=3, physics=False, supported=False):
    """Pose each controlled hinge: neutral -> lower -> upper -> neutral.

    With physics=True, command position actuators and integrate gravity/contact.
    Other joints receive neutral targets, but may move under load. Each joint's
    demonstration resets to the starting pose (especially useful without support).
    """
    if fps <= 0 or seconds_per_joint <= 0:
        raise ValueError("fps and seconds_per_joint must be positive")
    frames_per_leg = max(2, round(fps * seconds_per_joint / 3))
    data = mujoco.MjData(model)
    neutral = data.qpos.copy()
    joint_ids = list(dict.fromkeys(int(j) for j in model.actuator_trnid[:, 0]))
    for joint_id in joint_ids:
        if (model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
                or not model.jnt_limited[joint_id]):
            raise ValueError("This preview requires limited hinge joints")
        address = model.jnt_qposadr[joint_id]
        neutral[address] = np.clip(neutral[address], *model.jnt_range[joint_id])

    camera = make_camera()
    neutral_controls = neutral[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as video:
            for number, joint_id in enumerate(joint_ids, start=1):
                name = model.joint(joint_id).name
                address = model.jnt_qposadr[joint_id]
                lower, upper = model.jnt_range[joint_id]
                waypoints = [neutral[address], lower, upper, neutral[address]]
                if physics:
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = neutral
                    data.ctrl[:] = neutral_controls
                    mujoco.mj_forward(model, data)
                frame_number = 0
                print(f"{number}/{len(joint_ids)}: {name} "
                      f"({np.rad2deg(lower):.1f} to {np.rad2deg(upper):.1f} degrees)")
                for start, end in zip(waypoints[:-1], waypoints[1:]):
                    # Include both endpoints so the exact limits are shown.
                    for u in np.linspace(0, 1, frames_per_leg):
                        # In physics mode, hold each endpoint for the last 25%.
                        u = min(u / 0.75, 1.0) if physics else u
                        blend = u * u * (3 - 2 * u)  # Ease into/out of each limit.
                        angle = start + blend * (end - start)
                        if physics:
                            data.ctrl[:] = neutral_controls
                            actuator = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
                            data.ctrl[actuator] = angle
                            frame_number += 1
                            end_time = frame_number / fps
                            while data.time < end_time:
                                previous_time = data.time
                                mujoco.mj_step(model, data)
                                if data.time <= previous_time or not np.isfinite(data.qpos).all():
                                    raise RuntimeError("Physics became unstable during the joint sweep")
                        else:
                            data.qpos[:] = neutral
                            data.qpos[address] = angle
                            data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                        if physics:
                            camera.lookat[:] = data.body("torso").xpos
                        renderer.update_scene(data, camera=camera)
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 66), fill="black")
                        draw.text((10, 6), f"{number}/{len(joint_ids)}  {name}", fill="white")
                        draw.text((10, 25),
                                  f"Angle: {np.rad2deg(angle):.1f} deg   "
                                  f"Range: [{np.rad2deg(lower):.1f}, {np.rad2deg(upper):.1f}] deg",
                                  fill="white")
                        if physics:
                            draw.text((10, 44),
                                      f"Actual: {np.rad2deg(data.qpos[address]):.1f} deg | "
                                      f"Gravity ON | Torso {'supported' if supported else 'free'}",
                                      fill="yellow")
                        video.append_data(np.asarray(frame))
    print(f"Saved {output_path}")


def make_camera():
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.2]
    camera.distance = 4
    camera.azimuth = 135
    camera.elevation = -15
    return camera


def load_model(support_torso=False):
    path = Path(__file__).with_name("my_humanoid.xml")
    if not support_torso:
        return mujoco.MjModel.from_xml_path(str(path))
    # Temporary world weld acts as an external torso support. The saved XML
    # and the training model remain free to move. Gravity still acts on limbs.
    root = ET.parse(path).getroot()
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="preview_support", body1="torso")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def animate_kick(model, output_path="humanoid.mp4", fps=30, duration=5):
    """Render the original directly posed right-leg kick."""
    data = mujoco.MjData(model)
    hip_index = int(model.joint("right_hip_y").qposadr[0])
    knee_index = int(model.joint("right_knee").qposadr[0])
    key_times = np.array([0.0, 0.6, 0.9, 1.3, 2.0])
    hip_angles = np.deg2rad([0, -60, -70, -60, 0])
    knee_angles = np.deg2rad([-2, -90, -5, -90, -2])
    initial_pose = data.qpos.copy()
    initial_pose[int(model.joint("left_knee").qposadr[0])] = np.deg2rad(-2)
    camera = make_camera()
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as video:
            for frame in range(fps * duration):
                kick_time = (frame / fps) % 2.5
                data.qpos[:] = initial_pose
                data.qpos[hip_index] = np.interp(kick_time, key_times, hip_angles)
                data.qpos[knee_index] = np.interp(kick_time, key_times, knee_angles)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                video.append_data(renderer.render())
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview humanoid joint ranges or a kick.")
    parser.add_argument("--mode", choices=("ranges", "kick", "physics"), default="ranges")
    parser.add_argument("--free-root", action="store_true",
                        help="In physics mode, remove torso support; the humanoid can fall.")
    args = parser.parse_args()
    if args.free_root and args.mode != "physics":
        parser.error("--free-root requires --mode physics")
    supported = args.mode == "physics" and not args.free_root
    model = load_model(support_torso=supported)
    if args.mode == "physics":
        filename = "joint_ranges_gravity.mp4" if supported else "joint_ranges_free.mp4"
        animate_joint_ranges(model, output_path=filename, seconds_per_joint=6,
                             physics=True, supported=supported)
    elif args.mode == "ranges":
        animate_joint_ranges(model)
    else:
        animate_kick(model)
