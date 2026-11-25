"""Pupper + SO-101 ball-pick sim: the scripted grasp works and is physical."""

import pytest

from mjlab.asset_zoo.robots.pupper_v3_so101.pick_sim import RECOMMENDED_CROUCH, PickSim


@pytest.fixture(scope="module")
def sim() -> PickSim:
  return PickSim()


def test_sitting_pick_lifts_the_ball(sim):
  res = sim.pick(0.26, 0.0)
  assert res.ok, res.why
  # The jaw stops on the 40 mm ball instead of squeezing through it.
  assert res.grip_angle > 0.25
  assert res.grasp_error < 0.005
  # Sitting on its knees with the leg motors off, the body barely moves.
  assert res.base_move < 0.002
  assert res.leg_torque == 0.0


def test_crouched_pick_keeps_the_sled_off_the_floor(sim):
  res = sim.pick(0.26, 0.0, crouch=RECOMMENDED_CROUCH)
  assert res.ok, res.why
  assert res.sled_gap > 0.015
  assert res.leg_torque < 1.0


def test_ball_out_of_reach_is_not_picked(sim):
  res = sim.pick(0.40, 0.0)
  assert not res.ok
  assert res.why == "ik"
