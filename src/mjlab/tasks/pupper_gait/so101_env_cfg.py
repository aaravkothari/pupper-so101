"""Pupper v3 carrying an SO-101 arm: the VelocityFS task on the loaded robot.

The robot is ``pupper_v3_so101``: the arm rests unpowered (locked, folded) on a
printed back mount, and a USB-C power bank rides in a sled under the front belly. The
policy, its 48-dim actor frame and its 12 leg actions are VelocityFS's, so it
exports and deploys exactly like a plain Pupper policy.

On top of VelocityFS:

- The payload's box colliders end the episode when they touch the ground. The
  bank is the lowest part of the body (~72 mm up when standing), so this also
  teaches the policy not to drop its belly onto bumps.
- The power bank mass is randomized by +-10% around its 505 g spec (cable,
  electronics and print variation), and the bank's and arm links' COMs by +-1 cm
  each.
"""

from __future__ import annotations

import dataclasses
import math

from mjlab.asset_zoo.robots.pupper_v3_so101.pupper_so101_constants import (
  PAYLOAD_COLLISION_GEOMS,
  POWER_BANK_MASS,
  SO101_ARM_BODIES,
  get_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.pupper_gait.pupper_gait_env_cfg import pupper_velocity_fs_env_cfg
from mjlab.tasks.velocity.mdp.terminations import illegal_contact

# Power bank mass range (kg). pseudo_inertia scales mass by exp(2 * alpha).
POWER_BANK_MASS_RANGE: tuple[float, float] = (
  0.9 * POWER_BANK_MASS,
  1.1 * POWER_BANK_MASS,
)
PAYLOAD_COM_RANGE: float = 0.01  # m, per axis.
# The base task's density scale for every body, x[0.9, 1.3]; the arm event below
# replaces it for the arm links, so it repeats it.
_GLOBAL_ALPHA_RANGE: tuple[float, float] = (-0.0527, 0.1312)


def _alpha(mass: float) -> float:
  return 0.5 * math.log(mass / POWER_BANK_MASS)


def pupper_so101_velocity_fs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pupper_velocity_fs_env_cfg(play=play)

  # Swap the model only: the base cfg already wrapped the robot's actuators in
  # the deploy latency model, which must be kept.
  robot = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = dataclasses.replace(robot, spec_fn=get_spec)

  payload_pattern = "^(" + "|".join(PAYLOAD_COLLISION_GEOMS) + ")$"
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="payload_ground_contact",
      primary=ContactMatch(mode="geom", pattern=payload_pattern, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    ),
  )
  cfg.terminations["payload_ground_contact"] = TerminationTermCfg(
    func=illegal_contact,
    params={"sensor_name": "payload_ground_contact", "force_threshold": 1.0},
  )

  # Startup events run in insertion order and pseudo_inertia perturbs the
  # *default* inertials, so these replace the base task's "inertial" event for
  # the bodies they name.
  cfg.events["power_bank_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("power_bank",)),
      "alpha_range": tuple(_alpha(m) for m in POWER_BANK_MASS_RANGE),
      "t_range": (-PAYLOAD_COM_RANGE, PAYLOAD_COM_RANGE),
    },
  )
  cfg.events["arm_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=SO101_ARM_BODIES),
      "alpha_range": _GLOBAL_ALPHA_RANGE,
      "t_range": (-PAYLOAD_COM_RANGE, PAYLOAD_COM_RANGE),
    },
  )
  return cfg
