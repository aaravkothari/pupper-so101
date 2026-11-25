"""Watch Pupper + SO-101 pick ping-pong balls in the MuJoCo viewer.

  uv run python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_demo
  uv run python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_demo --crouch 0.03 0.02
  uv run python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_demo \\
    --balls 0.26,0 0.22,0.14

Each pick is planned and simulated first (pick_sim.PickSim.pick), recording the
motor commands; the viewer then replays those commands with live physics, in a
loop. By default the leg motors are off (Pupper sits on its knees). Pick
"so101_wrist_cam" under Rendering > Camera for the wrist camera view.
"""

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.asset_zoo.robots.pupper_v3_so101.pick_sim import PickSim

DEFAULT_BALLS = (
  (0.26, 0.0),
  (0.22, 0.14),
  (0.10, -0.20),
  (0.30, 0.06),
  (0.06, 0.22),
  (0.20, -0.10),
)
HOLD_SECONDS = 1.5


def main() -> None:
  parser = argparse.ArgumentParser(description="Pupper + SO-101 ball-pick demo.")
  parser.add_argument(
    "--crouch",
    type=float,
    nargs=2,
    metavar=("FRONT", "BACK"),
    help="Powered crouch (foot raise, m) instead of sitting with the legs off.",
  )
  parser.add_argument(
    "--balls",
    nargs="+",
    metavar="X,Y",
    help="Ball positions, m forward,left of the body center.",
  )
  args = parser.parse_args()
  balls = (
    [tuple(map(float, b.split(","))) for b in args.balls]
    if args.balls
    else DEFAULT_BALLS
  )
  crouch = tuple(args.crouch) if args.crouch else None

  sim = PickSim()
  m = sim.m
  takes = []
  for x, y in balls:
    start: list[tuple[np.ndarray, ...]] = []
    ctrls: list[np.ndarray] = []

    def record(d: mujoco.MjData, start=start, ctrls=ctrls) -> None:
      # The state after the first step is where the replay starts; the commands
      # of every later step are replayed from there.
      if not start:
        start.append((d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()))
      else:
        ctrls.append(d.ctrl.copy())

    res = sim.pick(x, y, crouch=crouch, on_step=record)
    print(f"ball at ({x:.2f}, {y:+.2f}) m: {'picked' if res.ok else res.why}")
    if ctrls:
      ctrls += [ctrls[-1]] * int(HOLD_SECONDS / m.opt.timestep)
      takes.append((start[0], ctrls))
  if not takes:
    raise SystemExit("No reachable balls to show.")

  data = mujoco.MjData(m)
  state = {"take": 0, "k": 0}

  def begin(i: int) -> None:
    (qpos, qvel, ctrl), _ = takes[i]
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
    state.update(take=i, k=0)

  def control(model: mujoco.MjModel, d: mujoco.MjData) -> None:
    del model
    _, ctrls = takes[state["take"]]
    if state["k"] >= len(ctrls):
      begin((state["take"] + 1) % len(takes))
      return
    d.ctrl[:] = ctrls[state["k"]]
    state["k"] += 1

  begin(0)
  mujoco.mj_forward(m, data)
  mujoco.set_mjcb_control(control)
  mujoco.viewer.launch(m, data)


if __name__ == "__main__":
  main()
