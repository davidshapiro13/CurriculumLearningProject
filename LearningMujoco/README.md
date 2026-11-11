# Humanoid with 20 target-angle controls

`my_humanoid.xml` has a rigid torso, a free (uncontrolled) root, and on each
side: 3 hip, 1 knee, 2 ankle, 3 shoulder, and 1 elbow hinge rotations.
Shoulder1/2 retain the stock oblique axes; shoulder3 twists along the upper arm.
The feet are boxes attached at the shin ends so ankle rotations are visible.
Joint limits are starting design choices, not anatomical measurements.

The XML's joint ranges use degrees. Position actuator controls use **radians**
with gear 1. Gains and force limits are initial settings that need tuning;
this model does not automatically balance. The original Humanoid reward still
rewards forward motion and penalizes action magnitude (now target angles).
Adapt that reward for your eventual task.

## Action order

Indices 0–9 are right, 10–19 are left, in this order:

1. hip_x
2. hip_z
3. hip_y
4. knee
5. ankle_y
6. ankle_x
7. shoulder1
8. shoulder2
9. shoulder3
10. elbow

Do not infer action order from joint IDs or qpos order. Use actuator names.

```python
from pathlib import Path
import gymnasium as gym
import numpy as np

# When running from the project root:
env = gym.make(
    "Humanoid-v5",
    xml_file=str(Path("LearningMujoco/my_humanoid.xml").resolve()),
)
observation, info = env.reset(seed=0)
# Replace this with your policy's 20 absolute target angles in degrees.
target_degrees = np.zeros(20)
target_degrees[[3, 13]] = -2  # Knees: nearly straight.
action = np.clip(
    np.deg2rad(target_degrees), env.action_space.low, env.action_space.high
)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

These are absolute targets, not changes to the previous angles. Actual angles
follow through forces and physics. Reset when terminated or truncated.

## Preview

From the project root, with the virtual environment activated:

```bash
python LearningMujoco/animate_humanoid.py
```

Writes `joint_ranges.mp4` to the working directory: each of the 20 controlled
hinges moves from neutral to its lower limit, to its upper limit, and back,
while other joints stay neutral. Each segment takes about three seconds and
is labeled with its joint name, angle, and limits. The free root is excluded.
Extreme poses may intersect other geometry because this preview sets poses directly.

Use `python LearningMujoco/animate_humanoid.py --mode kick` for the original
kick video (`humanoid.mp4`). Neither preview tests physical balance or
position-controller performance.

`generate_humanoid.py` recreates the custom model from the installed Gymnasium
stock asset. It refuses to overwrite an existing XML to protect manual edits.
Edit `my_humanoid.xml` directly; move it aside before regenerating intentionally.

## Preview with gravity and motor control

```bash
python LearningMujoco/animate_humanoid.py --mode physics
```

Writes `joint_ranges_gravity.mp4`. A temporary weld supports the torso at its
starting pose while gravity, contacts, joint limits, and position actuators are
simulated. Each joint receives a neutral -> lower -> upper -> neutral target
trajectory over six seconds, with short holds at the endpoints. The overlay
shows commanded and actual angles: actual motion may not reach the limits
because of load, collisions, or actuator strength. Other joints receive neutral
targets but can deflect under load. Every joint demonstration starts with a
state reset. The saved XML is unchanged.

```bash
python LearningMujoco/animate_humanoid.py --mode physics --free-root
```

Writes `joint_ranges_free.mp4` without support. The humanoid can fall; the
camera follows its torso. This demonstrates the need for balance control,
rather than guaranteeing that every joint can reach its full range.
