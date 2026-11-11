"""Create the 20-control model from Gymnasium's stock humanoid."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import gymnasium


def customize(root):
    root.set("model", "humanoid_20_angles")
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            if joint.get("name", "").startswith("abdomen_"):
                body.remove(joint)

    for side in ("right", "left"):
        foot = root.find(f".//body[@name='{side}_foot']")
        # Put the ankle at the shin end; elongated feet make rotation visible.
        foot.set("pos", "0 0 -0.3")
        for axis, direction, limits in (
            ("y", "0 1 0", "-45 30"),
            ("x", "1 0 0", "-20 20"),
        ):
            ET.SubElement(foot, "joint", name=f"{side}_ankle_{axis}",
                          type="hinge", axis=direction, range=limits,
                          damping="3", armature="0.01")
        geom = foot.find("geom")
        geom.attrib.clear()
        geom.attrib.update(name=f"{side}_foot", type="box", pos="0.04 0 -0.05",
                           size="0.12 0.055 0.04", condim="3", friction="1 0.1 0.1")
        arm = root.find(f".//body[@name='{side}_upper_arm']")
        # Third axis follows the upper arm's longitudinal direction.
        direction = "1 -1 -1" if side == "right" else "1 1 -1"
        arm.insert(2, ET.Element("joint", name=f"{side}_shoulder3", type="hinge",
                                axis=direction, range="-60 60", damping="2",
                                armature="0.006", stiffness="0"))

    default = root.find("default")
    default.remove(default.find("motor"))
    actuator = root.find("actuator")
    actuator.clear()
    actuator.append(ET.Comment(" Controls are absolute target angles in RADIANS, gear=1. Gains are initial values, not a tuned balance controller. "))
    for side in ("right", "left"):
        for suffix in ("hip_x", "hip_z", "hip_y", "knee", "ankle_y", "ankle_x",
                       "shoulder1", "shoulder2", "shoulder3", "elbow"):
            name = f"{side}_{suffix}"
            joint = root.find(f".//joint[@name='{name}']")
            limits = " ".join(f"{math.radians(float(v)):.10f}" for v in joint.get("range").split())
            leg = suffix.startswith(("hip", "knee"))
            ET.SubElement(actuator, "position", name=name, joint=name, gear="1",
                          kp="100" if leg else "40", kv="10" if leg else "4",
                          ctrllimited="true", ctrlrange=limits, forcelimited="true",
                          forcerange="-200 200" if leg else "-60 60")
    return root


if __name__ == "__main__":
    source = Path(gymnasium.__file__).parent / "envs/mujoco/assets/humanoid.xml"
    target = Path(__file__).with_name("my_humanoid.xml")
    tree = ET.ElementTree(customize(ET.parse(source).getroot()))
    ET.indent(tree, space="    ")
    # Refuse to overwrite an existing, possibly hand-edited model.
    with target.open("x") as output:
        output.write(ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    print(f"Created {target}")
