"""Pupper v3 + SO-101: the locked arm is a rigid payload on the stock Pupper."""

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots.pupper_v3.pupper_constants import JOINT_NAMES
from mjlab.asset_zoo.robots.pupper_v3_so101.pupper_so101_constants import (
  SO101_JOINT_NAMES,
  get_pupper_so101_robot_cfg,
  get_spec,
)
from mjlab.entity.entity import Entity
from mjlab.tasks.pupper_gait.so101_env_cfg import POWER_BANK_MASS_RANGE


def test_locked_arm_keeps_pupper_joints_and_actuators():
  model = Entity(get_pupper_so101_robot_cfg()).spec.compile()
  hinges = [model.joint(i).name for i in range(1, model.njnt)]
  assert hinges == list(JOINT_NAMES)
  assert model.nu == len(JOINT_NAMES)


def test_locking_matches_articulated_pose():
  pose = (0.3, -0.5, 0.7, 0.2, -1.0, 0.4)
  locked = get_spec(arm_pose=pose).compile()
  free = get_spec(lock_arm=False).compile()
  locked_data, free_data = mujoco.MjData(locked), mujoco.MjData(free)
  for name, angle in zip(SO101_JOINT_NAMES, pose, strict=True):
    free_data.qpos[free.joint(name).qposadr[0]] = angle
  mujoco.mj_forward(locked, locked_data)
  mujoco.mj_forward(free, free_data)
  for body in ("so101_wrist", "so101_gripper", "so101_moving_jaw"):
    a, b = locked.body(body).id, free.body(body).id
    np.testing.assert_allclose(locked_data.xpos[a], free_data.xpos[b], atol=1e-9)
    np.testing.assert_allclose(locked_data.xquat[a], free_data.xquat[b], atol=1e-9)
  assert np.isclose(locked.body_subtreemass[1], free.body_subtreemass[1])


def test_power_bank_mass_is_configurable():
  light = get_spec(power_bank_mass=0.2).compile()
  heavy = get_spec(power_bank_mass=0.5).compile()
  diff = heavy.body("power_bank").mass - light.body("power_bank").mass
  assert np.isclose(diff, 0.3)


def test_so101_task_randomizes_bank_and_ends_on_payload_contact():
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  torch.manual_seed(0)
  cfg = load_env_cfg("Mjlab-VelocityFS-Flat-Pupper-SO101-v3")
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    bank = env.sim.mj_model.body("robot/power_bank").id
    mass = env.sim.model.body_mass[:, bank]
    lo, hi = POWER_BANK_MASS_RANGE
    assert lo - 1e-6 <= mass.min() and mass.max() <= hi + 1e-6
    assert mass.max() - mass.min() > 0.02  # Actually randomized.
    assert env.action_manager.total_action_dim == 12  # Deploys like Pupper.

    env.reset()
    robot = env.scene["robot"]
    pose = robot.data.root_link_pose_w.clone()
    pose[:4, 2] = 0.05  # Belly on the floor: the sled is 68 mm below the base.
    robot.write_root_link_pose_to_sim(pose[:4], env_ids=torch.arange(4))
    env.step(torch.zeros(env.num_envs, 12))
    hit = env.termination_manager.get_term("payload_ground_contact")
    assert hit[:4].all()
    assert not hit[4:].any()
  finally:
    env.close()
