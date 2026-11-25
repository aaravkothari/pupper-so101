"""Pupper v3 carrying an SO-101 arm on a printed back mount.

The legs, actuators, collisions and simulation options are Pupper v3's
(``pupper_constants``), so every Pupper task and the deploy contract work
unchanged with this robot: the policy still sees and commands the 12 leg joints.

The arm's power (a USB-C PD power bank in a printed sled, plus the PD trigger
and 12 V buck) hangs under the front of the belly. The bank's mass is
``POWER_BANK_MASS``; Pupper's DeWalt battery stays.

The arm rests unpowered while walking. By default ``get_spec`` locks it: each of
its 6 joints is baked into the link frames at ``SO101_STOW_POSE`` and removed,
so the arm is a rigid payload with the right mass, COM and inertia, and a box
collider (``so101_arm_collision``) is fitted around it. Pass ``lock_arm=False``
for the articulated arm (for picking tasks; give it actuators in the task's
``EntityArticulationInfoCfg``).

Payload colliders are named ``*_collision``, so Pupper's ``CollisionCfg`` turns
them on: ``PAYLOAD_COLLISION_GEOMS`` lists them for contact sensors.
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  HOME_KEYFRAME,
  PUPPER_ARTICULATION,
  PUPPER_COLLISION,
)
from mjlab.entity import EntityCfg

##
# MJCF and assets.
##

PUPPER_SO101_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pupper_v3_so101"
  / "xmls"
  / "pupper_v3_so101.xml"
)
assert PUPPER_SO101_XML.exists()

SO101_JOINT_NAMES: tuple[str, ...] = (
  "so101_shoulder_pan",
  "so101_shoulder_lift",
  "so101_elbow_flex",
  "so101_wrist_flex",
  "so101_wrist_roll",
  "so101_gripper",
)

# Arm pose while walking, in SO101_JOINT_NAMES order (rad, so101_new_calib
# convention): the arm rests unpowered, the way it sits with the servos off.
# Found by letting the limp arm settle under gravity in sim (no actuators,
# collisions on): folded on itself with the shoulder against its -1.745 stop.
# It is stable there. The powered CAD L-pose (all zeros) is not: with the
# servos off, the arm falls forward over the nose from it.
SO101_STOW_POSE: tuple[float, ...] = (0.0, -1.744, 1.583, 0.875, 0.0, 0.0)

# Arm power bank in the sled (kg): UGREEN 145 W 25000 mAh (90597A), 160 x 80 x
# 26 mm, 505 g by its spec sheet. Weigh yours (with cable) and replace this.
POWER_BANK_MASS: float = 0.505


# Box colliders on the payload (the arm's is added by get_spec when locked).
PAYLOAD_COLLISION_GEOMS: tuple[str, ...] = (
  "so101_arm_collision",
  "battery_sled_collision",
)

# Bodies of the locked arm (they carry its mass; the mounts are separate).
SO101_ARM_BODIES: tuple[str, ...] = (
  "so101_base",
  "so101_shoulder",
  "so101_upper_arm",
  "so101_lower_arm",
  "so101_wrist",
  "so101_gripper",
  "so101_moving_jaw",
  "so101_wrist_camera",
)


def _lock_joints(spec: mujoco.MjSpec, pose: dict[str, float]) -> None:
  """Bake hinge angles into their bodies' frames and delete the joints."""
  for body in spec.bodies:
    for joint in list(body.joints):
      if joint.name not in pose:
        continue
      if joint.type != mujoco.mjtJoint.mjJNT_HINGE or np.any(joint.pos != 0):
        raise ValueError(f"Can only lock hinges at the body origin: {joint.name}")
      axis = np.asarray(joint.axis, dtype=float)
      rot = np.zeros(4)
      mujoco.mju_axisAngle2Quat(rot, axis / np.linalg.norm(axis), pose[joint.name])
      quat = np.zeros(4)
      mujoco.mju_mulQuat(quat, np.asarray(body.quat, dtype=float), rot)
      body.quat = quat
      spec.delete(joint)


def get_spec(
  lock_arm: bool = True,
  arm_pose: tuple[float, ...] = SO101_STOW_POSE,
  power_bank_mass: float = POWER_BANK_MASS,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PUPPER_SO101_XML))
  spec.geom("power_bank").mass = power_bank_mass
  if lock_arm:
    _lock_joints(spec, dict(zip(SO101_JOINT_NAMES, arm_pose, strict=True)))
    _add_arm_collider(spec)
  return spec


def _add_arm_collider(spec: mujoco.MjSpec) -> None:
  """Fit a base_link-frame box around the (locked) arm's meshes."""
  # Measure on a copy: compiling caches the spec's name lookup, which would hide
  # the geom added below from later spec.geom(name) calls (e.g. CollisionCfg).
  model = spec.copy().compile()
  data = mujoco.MjData(model)
  data.qpos[3] = 1.0  # Base at the origin, identity orientation.
  mujoco.mj_kinematics(model, data)
  arm = {model.body(name).id for name in SO101_ARM_BODIES}
  points = []
  for g in range(model.ngeom):
    if (
      model.geom_bodyid[g] not in arm
      or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
    ):
      continue
    mesh = model.geom_dataid[g]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    verts = model.mesh_vert[start : start + count]
    points.append(verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
  pts = np.concatenate(points) - data.xpos[model.body("base_link").id]
  lo, hi = pts.min(axis=0), pts.max(axis=0)
  spec.body("base_link").add_geom(
    name="so101_arm_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(lo + hi) / 2,
    size=(hi - lo) / 2,
    group=3,
    mass=0.0,
    density=0.0,
  )


##
# Final config.
##


def get_pupper_so101_robot_cfg() -> EntityCfg:
  """Pupper v3 + SO-101 with the arm locked in its stow pose."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(PUPPER_COLLISION,),
    spec_fn=get_spec,
    articulation=PUPPER_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
    DEFAULT_POSE,
    PUPPER_MUJOCO_OPTIONS,
  )
  from mjlab.entity.entity import Entity

  # Standing on a floor with the legs holding the default pose (keyframe "home").
  spec = Entity(get_pupper_so101_robot_cfg()).spec
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=(2, 2, 0.1), contype=1, conaffinity=1
  )
  spec.worldbody.add_light(pos=(0, 0, 2), dir=(0, 0, -1))
  spec.option.timestep = PUPPER_MUJOCO_OPTIONS["timestep"]
  model = spec.compile()
  qpos = np.zeros(model.nq)
  qpos[2], qpos[3], qpos[7:] = HOME_KEYFRAME.pos[2], 1.0, DEFAULT_POSE
  spec.add_key(name="home", qpos=qpos.tolist(), ctrl=list(DEFAULT_POSE))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  viewer.launch(model, data)
