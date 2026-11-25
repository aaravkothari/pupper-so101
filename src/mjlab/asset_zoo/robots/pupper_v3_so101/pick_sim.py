"""Can Pupper + SO-101 pick a ping-pong ball off the floor? A scripted pick sim.

Not a training task: a plain MuJoCo model and a scripted grasp, to map where the
arm can reach and pick from a given leg pose. The model is ``get_spec(lock_arm=
False)`` plus:

- Pupper's leg position servos (kp 5.5, kd 0.25, +-3 Nm), or no leg torque at all
  (``legs_off``: Pupper folds down onto its knees, as with the motors off).
- STS3215-like arm position servos (kp 100, +-2.9 Nm; the gripper capped at 1 Nm,
  ~14 N of squeeze) and convex colliders on the two fingers.
- A 40 mm, 2.7 g ball, and a manipulation-grade solver (1 ms, implicitfast,
  elliptic cones, no-slip). Two settings matter: hard ball/finger contacts (with
  MuJoCo's default, mass-relative softness the jaws squeeze through a 2.7 g
  ball) and no-slip (without it the feet creep outward under load).

A pick: unfold the arm, plan a collision-free grasp (arm IK, checked against the
body, sled, legs and floor), servo down onto the ball re-solving IK every 50 ms
at the robot's current pose (as a camera-guided pick would), close, lift 7 cm and
hold for 1 s. It succeeds if the ball is still in the gripper.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import (
  DEFAULT_POSE,
  FOOT_SITE_NAMES,
  JOINT_NAMES,
  PUPPER_EFFORT_LIMIT,
  PUPPER_KD,
  PUPPER_KP,
)
from mjlab.asset_zoo.robots.pupper_v3_so101.pupper_so101_constants import (
  SO101_ARM_BODIES,
  SO101_JOINT_NAMES,
  SO101_STOW_POSE,
  get_spec,
)

BALL_RADIUS = 0.020
BALL_MASS = 0.0027
ARM_KP = 100.0
ARM_FORCE = 2.9  # Nm, STS3215 at 12 V (~30 kg.cm).
GRIP_FORCE = 1.0  # Nm on the jaw: ~14 N squeeze at the fingertips.
GRIP_OPEN = 0.75
GRIP_CLOSE = -0.17
BALL_SOLREF = (0.004, 1.0)
BALL_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
# Grasp point in the gripper body frame: between the fingers, 5 mm clear of the
# fixed finger's inner face, 15 mm up from its tip (so the tips clear the floor).
GRASP_POINT = np.array([0.016, 0.0, -0.090])
LIFT_HEIGHT = 0.07
# Powered crouch with the best reach that keeps the sled clear (>= 26 mm while
# picking, with the wrist camera): front feet raised 3 cm, back feet 2 cm. See
# pick_sweep.py. Pass crouch=None to sit with the leg motors off instead.
RECOMMENDED_CROUCH: tuple[float, float] = (0.03, 0.02)

_IK_SEEDS = (
  (0.0, 1.6, -0.8, 0.5, 0.0),
  (0.0, 1.2, -0.2, 1.0, 0.0),
  (0.0, 0.6, 0.4, 1.3, 0.0),
  (0.0, 1.7, -1.2, 0.0, 1.57),
  (0.0, 0.0, 0.0, 0.0, 0.0),
)
_APPROACH_TILTS = np.radians([0.0, 20.0, 40.0, 60.0])
_UNFOLDED = np.array([0.0, 0.0, 0.0, 0.0, 0.0, GRIP_OPEN])


def _geom_verts(m: mujoco.MjModel, g: int) -> np.ndarray:
  """Mesh vertices of geom g in its body's frame."""
  mid = m.geom_dataid[g]
  a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, m.geom_quat[g])
  return m.mesh_vert[a : a + n] @ rot.reshape(3, 3).T + m.geom_pos[g]


def _add_finger_colliders(spec: mujoco.MjSpec) -> None:
  """Convex hulls of the two fingers (the jaw meshes minus hub and servo)."""
  ref = spec.copy().compile()
  gripper, jaw = ref.body("so101_gripper").id, ref.body("so101_moving_jaw").id
  fixed = next(
    g
    for g in range(ref.ngeom)
    if ref.geom_bodyid[g] == gripper and "follower" in ref.mesh(ref.geom_dataid[g]).name
  )
  moving = next(g for g in range(ref.ngeom) if ref.geom_bodyid[g] == jaw)
  fv, jv = _geom_verts(ref, fixed), _geom_verts(ref, moving)
  r = np.linalg.norm(jv[:, :2], axis=1)  # Distance from the jaw pivot.
  pieces = {
    "so101_gripper": (
      fv[(fv[:, 2] < -0.075) & (fv[:, 0] < -0.006)],
      fv[(fv[:, 2] >= -0.078) & (fv[:, 2] < -0.045) & (fv[:, 0] < -0.006)],
    ),
    "so101_moving_jaw": (jv[r > 0.055], jv[(r > 0.025) & (r <= 0.058)]),
  }
  for body, parts in pieces.items():
    for k, pts in enumerate(parts):
      hull = trimesh.convex.convex_hull(pts)
      name = f"{body}_finger{k}"
      spec.add_mesh(
        name=name,
        uservert=hull.vertices.flatten().tolist(),
        userface=hull.faces.flatten().tolist(),
      )
      spec.body(body).add_geom(
        name=f"{name}_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        contype=2,
        conaffinity=2,
        condim=4,
        friction=(0.5, 0.01, 0.001),
        solref=BALL_SOLREF,
        solimp=BALL_SOLIMP,
        group=3,
        mass=0.0,
        density=0.0,
        rgba=(1, 0, 0, 0.4),
      )


def _add_position_servo(
  spec: mujoco.MjSpec, joint: str, kp: float, kd: float, limit: float
) -> None:
  spec.add_actuator(
    name=joint,
    target=joint,
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
    gaintype=mujoco.mjtGain.mjGAIN_FIXED,
    gainprm=[kp] + [0.0] * 9,
    biastype=mujoco.mjtBias.mjBIAS_AFFINE,
    biasprm=[0.0, -kp, -kd] + [0.0] * 7,
    forcelimited=1,
    forcerange=(-limit, limit),
  )


def build_model(timestep: float = 0.001) -> mujoco.MjModel:
  """Pupper + articulated SO-101 with servos, finger colliders, floor and ball."""
  spec = get_spec(lock_arm=False)
  _add_finger_colliders(spec)
  for joint in JOINT_NAMES:
    _add_position_servo(spec, joint, PUPPER_KP, PUPPER_KD, PUPPER_EFFORT_LIMIT)
  for joint in SO101_JOINT_NAMES:
    limit = GRIP_FORCE if joint == "so101_gripper" else ARM_FORCE
    _add_position_servo(spec, joint, ARM_KP, 1.0, limit)
  # Pupper's colliders (feet, knees, sled, ...) against the floor and ball.
  for g in spec.geoms:
    if g.name.endswith("_collision") and "finger" not in g.name:
      g.contype, g.conaffinity, g.condim = 1, 1, 3
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(3, 3, 0.1),
    contype=3,
    conaffinity=3,
    condim=3,
    friction=(0.8, 0.02, 0.01),
    rgba=(0.9, 0.9, 0.9, 1),
  )
  ball = spec.worldbody.add_body(name="ball", pos=(1, 1, BALL_RADIUS))
  ball.add_freejoint(name="ball")
  ball.add_geom(
    name="ball",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS, 0, 0),
    mass=BALL_MASS,
    contype=2,
    conaffinity=3,
    condim=4,
    friction=(0.5, 0.01, 0.001),
    solref=BALL_SOLREF,
    solimp=BALL_SOLIMP,
    rgba=(1, 0.55, 0.1, 1),
  )
  spec.worldbody.add_light(pos=(0, 0, 2), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8))
  spec.worldbody.add_light(
    pos=(0.5, -0.5, 0.6), dir=(-1, 1, -1), diffuse=(0.4, 0.4, 0.4)
  )
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 1200
  spec.option.timestep = timestep
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10
  spec.option.iterations = 50
  spec.option.noslip_iterations = 10
  return spec.compile()


@dataclass
class PickResult:
  ok: bool
  why: str = ""  # Failure reason: ik, body, sled/legs, path, dropped, ...
  tilt_deg: float = 0.0  # Approach tilt from vertical.
  leg_torque: float = 0.0  # Peak |leg actuator torque|, Nm.
  sled_gap: float = 1.0  # Lowest sled-to-floor distance, m.
  base_move: float = 0.0  # Largest base displacement during the pick, m.
  tilt_change_deg: float = 0.0  # Largest base rotation during the pick.
  grasp_error: float = 0.0  # Grasp point to ball center when closing, m.
  grip_angle: float = 0.0  # Jaw angle once closed on the ball, rad.
  extra: dict = field(default_factory=dict)


class PickSim:
  """The pick model, its data and the scripted pick."""

  def __init__(self, model: mujoco.MjModel | None = None):
    m = model if model is not None else build_model()
    self.m, self.d = m, mujoco.MjData(m)
    self.leg_q = np.array([m.joint(n).qposadr[0] for n in JOINT_NAMES])
    self.leg_v = np.array([m.joint(n).dofadr[0] for n in JOINT_NAMES])
    self.arm_q = np.array([m.joint(n).qposadr[0] for n in SO101_JOINT_NAMES])
    self.arm_v = np.array([m.joint(n).dofadr[0] for n in SO101_JOINT_NAMES])
    self.leg_act = np.array([m.actuator(n).id for n in JOINT_NAMES])
    self.arm_act = np.array([m.actuator(n).id for n in SO101_JOINT_NAMES])
    arm_jnt = [m.joint(n).id for n in SO101_JOINT_NAMES]
    self.arm_lo, self.arm_hi = m.jnt_range[arm_jnt, 0], m.jnt_range[arm_jnt, 1]
    self.feet = [m.site(n).id for n in FOOT_SITE_NAMES]
    self.gripper = m.body("so101_gripper").id
    self.shoulder = m.body("so101_shoulder").id
    self.ball_q = m.joint("ball").qposadr[0]
    self.ball_v = m.joint("ball").dofadr[0]
    self.sled = m.geom("battery_sled_collision").id
    self.floor = m.geom("floor").id
    self._leg_gain = m.actuator_gainprm[self.leg_act].copy()
    self._leg_bias = m.actuator_biasprm[self.leg_act].copy()
    self._limp_state: tuple[np.ndarray, ...] | None = None
    arm_bodies = {m.body(n).id for n in SO101_ARM_BODIES}
    self.arm_geoms = [
      g
      for g in range(m.ngeom)
      if m.geom_bodyid[g] in arm_bodies
      and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      and m.geom_group[g] == 1
    ]
    self.obstacles = [self.sled, m.geom("power_bank").id] + [
      g
      for g in range(m.ngeom)
      if m.body(m.geom_bodyid[g]).name.startswith("leg_")
      and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    # The torso is concave (its convex hull covers the arm mount), so the arm
    # is checked against a filled voxel model of it instead.
    self.body_voxels = []
    for g in range(m.ngeom):
      if m.geom_bodyid[g] != 1 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
      mid = m.geom_dataid[g]
      fa, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
      mesh = trimesh.Trimesh(_geom_verts(m, g), m.mesh_face[fa : fa + nf])
      self.body_voxels.append(mesh.voxelized(0.003).fill())

  # Legs.

  def crouch_pose(self, dz_front: float, dz_back: float) -> np.ndarray:
    """Leg joints that raise the feet by dz (base frame), abduction unchanged."""
    m, d = self.m, self.d
    q = np.array(DEFAULT_POSE, dtype=float)
    d.qpos[:] = 0
    d.qpos[3] = 1
    d.qpos[self.leg_q] = q
    mujoco.mj_kinematics(m, d)
    targets = [d.site_xpos[s].copy() for s in self.feet]
    for i, t in enumerate(targets):
      t[2] += dz_front if i < 2 else dz_back
    jac = np.zeros((3, m.nv))
    for _ in range(200):
      d.qpos[self.leg_q] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      worst = 0.0
      for i, s in enumerate(self.feet):
        err = targets[i] - d.site_xpos[s]
        worst = max(worst, float(np.linalg.norm(err)))
        mujoco.mj_jacSite(m, d, jac, None, s)
        cols = self.leg_v[3 * i + 1 : 3 * i + 3]  # Hip, knee.
        dq = np.linalg.lstsq(jac[[0, 2]][:, cols], err[[0, 2]], rcond=None)[0]
        q[3 * i + 1 : 3 * i + 3] += np.clip(dq, -0.2, 0.2)
      if worst < 1e-5:
        break
    return q

  def legs_off(self, off: bool = True) -> None:
    """Cut (or restore) leg motor torque; joint friction, damping, stops remain."""
    self.m.actuator_gainprm[self.leg_act] = 0 if off else self._leg_gain
    self.m.actuator_biasprm[self.leg_act] = 0 if off else self._leg_bias

  def stand(self, leg_pose: np.ndarray, seconds: float = 1.5) -> None:
    """Reset onto the floor in leg_pose (arm stowed) and let the servos settle."""
    m, d = self.m, self.d
    self.legs_off(False)
    mujoco.mj_resetData(m, d)
    d.qpos[3] = 1
    d.qpos[self.leg_q] = leg_pose
    d.qpos[self.arm_q] = SO101_STOW_POSE
    d.qpos[self.ball_q : self.ball_q + 4] = (1.0, 1.0, BALL_RADIUS, 1.0)
    d.ctrl[self.leg_act] = leg_pose
    d.ctrl[self.arm_act] = SO101_STOW_POSE
    mujoco.mj_kinematics(m, d)
    d.qpos[2] = 0.012 - min(d.site_xpos[s][2] for s in self.feet)
    for _ in range(int(seconds / m.opt.timestep)):
      mujoco.mj_step(m, d)

  def sit(self) -> None:
    """Stand, cut the leg motors and let Pupper fold onto its knees (cached)."""
    m, d = self.m, self.d
    if self._limp_state is None:
      self.stand(np.array(DEFAULT_POSE, dtype=float), seconds=1.0)
      self.legs_off(True)
      for _ in range(int(4.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
      self._limp_state = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy())
    self.legs_off(True)
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:], d.ctrl[:] = self._limp_state
    mujoco.mj_forward(m, d)

  # Arm.

  def grasp_point(self) -> np.ndarray:
    d = self.d
    return d.xpos[self.gripper] + d.xmat[self.gripper].reshape(3, 3) @ GRASP_POINT

  def arm_ik(
    self, pos: np.ndarray, approach: np.ndarray, seed: np.ndarray, iters: int = 300
  ) -> tuple[np.ndarray, bool]:
    """Grasp point at pos with the fingers along approach, side by side.

    Position comes first: the arm has 5 joints, so the approach direction is a
    soft goal (accepted within 12 degrees; the ball is a sphere).
    """
    m, d = self.m, self.d
    q = np.array(seed, dtype=float)
    approach = approach / np.linalg.norm(approach)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    qsave = d.qpos.copy()
    e_p = e_dir = np.ones(3)
    for _ in range(iters):
      d.qpos[self.arm_q] = q
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      rot = d.xmat[self.gripper].reshape(3, 3)
      p = d.xpos[self.gripper] + rot @ GRASP_POINT
      e_p = pos - p
      e_dir = np.cross(-rot[:, 2], approach)
      e_r = e_dir.copy()
      side = np.cross(approach, [0.0, 0.0, 1.0])
      if np.linalg.norm(side) > 1e-3:  # Fingers side by side: gripper x level.
        side /= np.linalg.norm(side)
        side = side if rot[:, 0] @ side >= 0 else -side
        e_r += 0.5 * np.cross(rot[:, 0], side)
      if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_dir) < 0.01:
        break
      mujoco.mj_jac(m, d, jp, jr, p, self.gripper)
      jac = np.vstack([jp[:, self.arm_v[:5]], 0.08 * jr[:, self.arm_v[:5]]])
      err = np.concatenate([e_p, 0.08 * e_r])
      dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
      q[:5] = np.clip(
        q[:5] + np.clip(dq, -0.15, 0.15), self.arm_lo[:5], self.arm_hi[:5]
      )
    d.qpos[:] = qsave
    mujoco.mj_kinematics(m, d)
    ok = np.linalg.norm(e_p) < 2e-3 and np.linalg.norm(e_dir) < np.sin(np.radians(12))
    return q, bool(ok)

  def arm_blocked(self, arm_pose: np.ndarray, ground: float = 0.0) -> str | None:
    """Why arm_pose collides at the current base pose, or None if it is clear."""
    m, d = self.m, self.d
    qsave = d.qpos.copy()
    d.qpos[self.arm_q] = arm_pose
    mujoco.mj_kinematics(m, d)
    base, base_rot = d.xpos[1].copy(), d.xmat[1].reshape(3, 3).copy()
    scratch = np.zeros(6)
    reason = None
    for g in self.arm_geoms:
      mid = m.geom_dataid[g]
      a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
      v = m.mesh_vert[a : a + n : 2] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
      if v[:, 2].min() < ground:
        reason = "ground"
        break
      if m.body(m.geom_bodyid[g]).name in ("so101_base", "so101_shoulder"):
        continue  # Bolted to the mount on top of the torso.
      local = (v - base) @ base_rot
      if any(vox.is_filled(local).any() for vox in self.body_voxels):
        reason = "body"
        break
      if any(
        mujoco.mj_geomDistance(m, d, g, o, 0.002, scratch) < 0 for o in self.obstacles
      ):
        reason = "sled/legs"
        break
    d.qpos[:] = qsave
    mujoco.mj_kinematics(m, d)
    return reason

  def _approach(self, ball: np.ndarray, tilt: float) -> np.ndarray:
    """Down, tilted by `tilt` away from the shoulder."""
    u = ball[:2] - self.d.xpos[self.shoulder][:2]
    u = u / (np.linalg.norm(u) + 1e-9)
    return np.array([np.sin(tilt) * u[0], np.sin(tilt) * u[1], -np.cos(tilt)])

  def plan(self, ball: np.ndarray) -> dict | str:
    """Pre-grasp/grasp/lift arm poses that clear the robot, or why there are none."""
    reasons = []
    base = self.d.xpos[1]
    pan = np.arctan2(ball[1] - base[1], ball[0] - base[0] + 0.02)
    up = np.array([0.0, 0.0, 1.0])
    for tilt in _APPROACH_TILTS:
      approach = self._approach(ball, tilt)
      for seed in _IK_SEEDS:
        grasp, ok = self.arm_ik(ball, approach, np.r_[pan, seed[1:], GRIP_OPEN])
        if not ok:
          reasons.append("ik")
          continue
        grasp[5] = GRIP_OPEN
        blocked = self.arm_blocked(grasp, ground=0.0015)
        if blocked:
          reasons.append(blocked)
          continue
        pre, ok_pre = self.arm_ik(ball + 0.06 * up, approach, grasp)
        lift, ok_lift = self.arm_ik(ball + LIFT_HEIGHT * up, approach, grasp)
        pre[5] = lift[5] = GRIP_OPEN
        path = [_UNFOLDED + s * (pre - _UNFOLDED) for s in (0.25, 0.5, 0.75)]
        if not (ok_pre and ok_lift) or any(
          self.arm_blocked(q) for q in [pre, lift, *path]
        ):
          reasons.append("path")
          continue
        return dict(tilt=tilt, pre=pre, grasp=grasp, lift=lift)
    return Counter(reasons).most_common(1)[0][0] if reasons else "ik"

  def pick(
    self,
    x: float,
    y: float,
    crouch: tuple[float, float] | None = None,
    on_step: Callable[[mujoco.MjData], None] | None = None,
  ) -> PickResult:
    """Pick a ball at (x, y) m from the body center (forward, left) on the floor.

    crouch: (front, back) foot raise in m for a powered crouch, or None to sit
    with the leg motors off. on_step is called after every physics step.
    """
    m, d = self.m, self.d
    sitting = crouch is None
    if crouch is None:
      self.sit()
    else:
      self.stand(self.crouch_pose(*crouch))
    fwd = d.xmat[1].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0], 0.0])
    ball = d.xpos[1] + x * fwd + y * left
    ball[2] = BALL_RADIUS
    d.qpos[self.ball_q : self.ball_q + 7] = (*ball, 1, 0, 0, 0)
    d.qvel[self.ball_v : self.ball_v + 6] = 0
    mujoco.mj_forward(m, d)
    plan = self.plan(ball)
    if isinstance(plan, str):
      return PickResult(ok=False, why=plan)

    res = PickResult(ok=False, tilt_deg=float(np.degrees(plan["tilt"])))
    base0, rot0 = d.qpos[:3].copy(), d.xmat[1].reshape(3, 3).copy()
    scratch = np.zeros(6)

    def step() -> None:
      mujoco.mj_step(m, d)
      res.leg_torque = max(
        res.leg_torque, float(np.abs(d.actuator_force[self.leg_act]).max())
      )
      if round(d.time / m.opt.timestep) % 20 == 0:
        gap = mujoco.mj_geomDistance(m, d, self.sled, self.floor, 0.2, scratch)
        res.sled_gap = min(res.sled_gap, float(gap))
        res.base_move = max(res.base_move, float(np.linalg.norm(d.qpos[:3] - base0)))
        cos = (np.trace(rot0.T @ d.xmat[1].reshape(3, 3)) - 1) / 2
        res.tilt_change_deg = max(
          res.tilt_change_deg, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        )
      if on_step is not None:
        on_step(d)

    def move(target: np.ndarray, seconds: float) -> None:
      """Joint-space move with a smooth profile."""
      start = d.ctrl[self.arm_act].copy()
      n = int(seconds / m.opt.timestep)
      for i in range(n):
        s = min(1.0, (i + 1) / (0.8 * n))
        d.ctrl[self.arm_act] = start + (3 * s**2 - 2 * s**3) * (target - start)
        step()

    def servo(
      target: Callable[[float], np.ndarray],
      seconds: float,
      grip: Callable[[float], float],
      dt: float = 0.05,
    ) -> None:
      """Track a moving grasp-point target, re-solving IK at the current pose."""
      n = int(round(seconds / dt))
      for k in range(n):
        s = (k + 1) / n
        s = 3 * s**2 - 2 * s**3
        q, _ = self.arm_ik(target(s), approach, d.qpos[self.arm_q], iters=60)
        q[5] = grip(s)
        d.ctrl[self.arm_act] = q
        for _ in range(int(dt / m.opt.timestep)):
          step()

    def ball_now() -> np.ndarray:
      return d.qpos[self.ball_q : self.ball_q + 3].copy()

    move(_UNFOLDED, 1.0)
    move(plan["pre"], 1.2)
    approach = self._approach(ball_now(), plan["tilt"])
    servo(lambda s: ball_now() + [0, 0, 0.06 * (1 - s)], 1.2, lambda s: GRIP_OPEN)
    res.grasp_error = float(np.linalg.norm(self.grasp_point() - ball_now()))
    hold = self.grasp_point().copy()
    servo(lambda s: hold, 0.7, lambda s: GRIP_OPEN + s * (GRIP_CLOSE - GRIP_OPEN))
    res.grip_angle = float(d.qpos[self.arm_q[5]])
    servo(lambda s: hold + [0, 0, LIFT_HEIGHT * s], 1.2, lambda s: GRIP_CLOSE)
    servo(lambda s: hold + [0, 0, LIFT_HEIGHT], 1.0, lambda s: GRIP_CLOSE)

    lifted = ball_now()[2] > BALL_RADIUS + 0.04
    held = np.linalg.norm(ball_now() - self.grasp_point()) < 0.03
    upright = d.xmat[1].reshape(3, 3)[2, 2] > 0.8
    sled_ok = sitting or res.sled_gap > 0.0
    res.ok = bool(lifted and held and upright and sled_ok)
    if not res.ok:
      res.why = (
        "tipped"
        if not upright
        else "sled hit floor"
        if not sled_ok
        else "dropped"
        if not lifted
        else "not held"
      )
    return res
