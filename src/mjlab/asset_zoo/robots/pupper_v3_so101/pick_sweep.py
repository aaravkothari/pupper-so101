"""Map where Pupper + SO-101 can pick a ping-pong ball, for one or more leg poses.

  uv run python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_sweep --out sweep.json
  uv run python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_sweep \\
    --crouch 0.03 0.01 --crouch 0.05 0.05 --out crouch.json
  uv run --with matplotlib python -m mjlab.asset_zoo.robots.pupper_v3_so101.pick_sweep \\
    --plot reach.png

Runs one full simulated pick (pick_sim.PickSim.pick) per 2 cm grid spot on the
floor, in parallel, and prints a map per leg pose. Without --crouch the leg
motors are off (Pupper sits on its knees). --plot draws the maps (needs
matplotlib).

Results so far (40 mm ball, wrist camera on, 475-spot grid):

- Sitting with the leg motors off: 240 picked, 4-32 cm ahead and out to 24 cm
  to the sides; the body moves <= 0.5 mm during a pick.
- Crouch front +3 / back +2 cm (RECOMMENDED_CROUCH): 186 picked, reach 30 cm,
  <= 0.72 Nm on any leg, sled >= 26 mm off the floor, body moves <= 8.5 mm.
- Level crouch +5 cm: 211 picked but the sled comes within 2 mm of the floor;
  front +3 / back +1 cm pitches onto the sled.
"""

import argparse
import dataclasses
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np

from mjlab.asset_zoo.robots.pupper_v3_so101.pick_sim import PickSim

XS = np.round(np.arange(0.04, 0.401, 0.02), 3)
YS = np.round(np.arange(-0.24, 0.241, 0.02), 3)
_SYMBOLS = {"": "#", "ik": ".", "body": "b", "sled/legs": "s", "path": "p"}

_sim: PickSim | None = None


def _init_worker() -> None:
  global _sim
  _sim = PickSim()


def _run(job: tuple[tuple[float, float] | None, float, float]) -> dict:
  crouch, x, y = job
  assert _sim is not None
  res = _sim.pick(x, y, crouch=crouch)
  return dict(crouch=crouch, x=float(x), y=float(y), **dataclasses.asdict(res))


def sweep(crouches: list[tuple[float, float] | None], workers: int) -> list[dict]:
  jobs = [(c, x, y) for c in crouches for x in XS for y in YS]
  with Pool(workers, initializer=_init_worker) as pool:
    return pool.map(_run, jobs, chunksize=8)


def _label(crouch: tuple[float, float] | None) -> str:
  if crouch is None:
    return "leg motors off (sitting)"
  return f"crouch: front feet +{crouch[0] * 100:.0f} cm, back +{crouch[1] * 100:.0f} cm"


def print_maps(results: list[dict]) -> None:
  """One text map per leg pose: # picked, . out of reach, b/s/p blocked, x lost."""
  by_pose: dict = {}
  for r in results:
    by_pose.setdefault(tuple(r["crouch"]) if r["crouch"] else None, []).append(r)
  for crouch, rs in by_pose.items():
    ok = [r for r in rs if r["ok"]]
    fails = Counter(r["why"] for r in rs if not r["ok"])
    print(f"\n{_label(crouch)}: {len(ok)}/{len(rs)} picked; failures {dict(fails)}")
    if ok:
      print(
        f"  peak leg torque {max(r['leg_torque'] for r in ok):.2f} Nm, "
        f"body moved <= {max(r['base_move'] for r in ok) * 1000:.1f} mm"
      )
    grid = {(r["x"], r["y"]): r for r in rs}
    print("  y (cm): " + "".join(f"{y * 100:+4.0f}" for y in YS))
    for x in XS[::-1]:
      row = "".join(
        f"{_SYMBOLS.get(grid[(x, y)]['why'], 'x'):>4}" for y in YS if (x, y) in grid
      )
      print(f"  x {x * 100:3.0f} cm{row}")


def plot_maps(results: list[dict], path: str) -> None:
  """Side-by-side maps, one per leg pose. Needs matplotlib."""
  import matplotlib  # Optional dependency: only for --plot.

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Patch, Rectangle

  colors = {"ok": "#2a9d55", "ik": "#e6e6e6", "blocked": "#e9a23b", "lost": "#c0392b"}
  by_pose: dict = {}
  for r in results:
    by_pose.setdefault(tuple(r["crouch"]) if r["crouch"] else None, []).append(r)
  fig, axes = plt.subplots(
    1, len(by_pose), figsize=(6 * len(by_pose), 6.5), squeeze=False
  )
  for ax, (crouch, rs) in zip(axes[0], by_pose.items(), strict=True):
    ax.add_patch(Rectangle((-0.108, -0.061), 0.259, 0.122, fc="#bbb", ec="#555"))
    ax.add_patch(Rectangle((0.020, -0.044), 0.162, 0.087, fc="#1a9e8a", ec="#0d5f53"))
    for r in rs:
      kind = (
        "ok"
        if r["ok"]
        else "ik"
        if r["why"] == "ik"
        else "blocked"
        if r["why"] in ("body", "sled/legs", "path")
        else "lost"
      )
      ax.add_patch(
        Rectangle(
          (r["x"] - 0.009, r["y"] - 0.009),
          0.018,
          0.018,
          fc=colors[kind],
          zorder=0 if kind == "ik" else 3,
        )
      )
    ok = sum(r["ok"] for r in rs)
    ax.set_title(f"{_label(crouch)}\n{ok}/{len(rs)} spots picked", loc="left")
    ax.set_xlim(-0.13, 0.42)
    ax.set_ylim(-0.26, 0.26)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("forward from body center (m)")
  axes[0][0].set_ylabel("left of centerline (m)")
  legend = [
    Patch(fc=colors["ok"], label="picked up"),
    Patch(fc=colors["blocked"], label="arm would hit the robot"),
    Patch(fc=colors["lost"], label="grasped, then lost"),
    Patch(fc=colors["ik"], label="out of reach"),
    Patch(fc="#bbb", ec="#555", label="body"),
    Patch(fc="#1a9e8a", label="sled"),
  ]
  fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False)
  fig.tight_layout(rect=(0, 0.06, 1, 1))
  fig.savefig(path, dpi=110)
  print(f"wrote {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Pupper + SO-101 ball-pick reach map.")
  parser.add_argument(
    "--crouch",
    type=float,
    nargs=2,
    action="append",
    metavar=("FRONT", "BACK"),
    help="A powered crouch to map (foot raise, m); repeatable. Default: legs off.",
  )
  parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
  parser.add_argument("--out", help="Write every pick result to this JSON file.")
  parser.add_argument("--plot", help="Draw the maps to this image (needs matplotlib).")
  args = parser.parse_args()
  crouches: list[tuple[float, float] | None] = (
    [(c[0], c[1]) for c in args.crouch] if args.crouch else [None]
  )
  results = sweep(crouches, args.workers)
  print_maps(results)
  if args.out:
    with open(args.out, "w") as f:
      json.dump(results, f)
  if args.plot:
    plot_maps(results, args.plot)


if __name__ == "__main__":
  main()
