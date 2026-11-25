"""Regenerate the mesh assets of the Pupper v3 + SO-101 model.

Run this again whenever the printed mounts change (re-export the CadQuery model
first):

  uv run --with fast-simplification python -m \\
    mjlab.asset_zoo.robots.pupper_v3_so101.build_assets \\
    --mount-dir <pupper_so101_mount_v3>/out \\
    --saddle-print partA_saddle_print_v4.stl \\
    --sled sled.stl --sled-floor sled_floor.stl \\
    --so101-dir <SO-ARM100>/Simulation/SO101/assets \\
    --wrist-cam-mount <SO-ARM100>/Optional/Wrist_Cam_Mount_32x32_UVC_Module/stl/Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl

Inputs:
- ``--mount-dir``: ``saddle_pupper_frame.stl`` and ``arm_base_pupper_frame.stl``,
  exported by ``pupper_so101_mount.py`` in the Pupper STEP frame (mm; +Y up,
  nose toward -Z, X lateral).
- ``--saddle-print`` (optional): a newer saddle in print orientation (e.g. v4,
  which only ships as a print STL). It is put back in the Pupper frame by
  undoing ``pupper_so101_mount.py``'s ``export`` (rotate -90 deg about X, then
  center the bounding box and drop it onto z = 0), with the offset taken from
  ``saddle_pupper_frame.stl``: valid while the saddle's outline is unchanged.
- ``--sled`` / ``--sled-floor``: the power-bank sled and its floor plate, in
  print orientation (mm; sled base on z = 0, open side +z, closed end -y). On
  the robot the base is pushed flat against the belly by 4 push pins through
  its holes into the belly's vent holes (the same 4 as pupper_belly_cradle.py),
  the open side faces down, and the floor slides in from the open end onto the
  inward lips, carrying the bank. The sled is placed from its own pin holes.
- ``--so101-dir``: the official SO-101 meshes from TheRobotStudio/SO-ARM100
  (Apache-2.0). The Hiwonder kit is the same geometry.

- ``--wrist-cam-mount``: SO-ARM100's one-piece wrist camera mount for a 32 x 32 mm
  UVC module (a Wrist_Roll_Follower with the camera plate built on). It is drawn
  in the follower's frame (mm), so it replaces the follower mesh 1:1. A mesh of
  the camera module (board + lens holder) is generated on its plate, toward the
  fingers.

The power bank is a box in the XML (``print_sled_xml`` prints its pose).

The mounts are rewritten in the MJCF ``base_link`` frame (m), so they sit at the
origin of their bodies. The SO-101 meshes are decimated (visual only) to keep
the repo small.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh

ASSET_DIR = Path(__file__).parent / "xmls" / "assets"

# Pupper STEP frame -> MJCF base_link frame, found by registering the MJCF body
# mesh (BodyV4v70_001) against Pupper_v3_5_v14.stl (ICP, 1.5 mm median error;
# the residual is the v3.5 vs V4 body difference).
#   base_link x (forward) = -STEP Z,  y (left) = -STEP X,  z (up) = STEP Y.
STEP_TO_BASE_ROT = np.array(
  [
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
  ]
)
# base_link origin expressed in the STEP frame (m).
BASE_ORIGIN_IN_STEP = np.array([0.00051, -0.06816, 0.00213])

# (source STL, output STL, printed mass in kg). Masses are the slicer estimates
# from the mount README; replace them with scale readings.
MOUNTS = (
  ("saddle_pupper_frame.stl", "pupper_so101_saddle.stl", 0.048),
  ("arm_base_pupper_frame.stl", "pupper_so101_arm_base.stl", 0.044),
)
# The README's 48 g is for v3's 80.8 cm^3; a --saddle-print scales it by volume.
SADDLE_V3_VOLUME_MM3 = 80_825.0
# pupper_so101_mount.py exports the saddle rotated -90 deg about X:
# (x, y, z) -> (x, z, -y).
SADDLE_PRINT_ROT = np.array(
  [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
  ]
)

# Sled, from pupper_belly_cradle.py (STEP frame, mm): the belly plate's
# underside, and the 4 vent holes its pins use, as (X, Z), matched to the sled's
# holes in the same order after sorting both by (Z, X).
BELLY_UNDERSIDE_Y = -99.75
BELLY_PIN_HOLES = np.array(
  [(-24.56, -115.08), (25.44, -115.08), (-24.56, -45.08), (25.44, -45.08)]
)
# Sled print frame -> STEP, with the closed end toward the nose: sled x -> X,
# sled y -> Z (rearward), sled z -> -Y (down). The base then ends ~1 mm in front
# of the DeWalt opening (Z = -21), like the cradle's plate.
SLED_TO_STEP_ROT = np.array(
  [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
  ]
)
# Printed masses (kg): the sled's 2.4 mm walls are solid perimeters (~1.1 g/cm^3
# of its 75.2 cm^3); the vented 2.5 mm floor is mostly top/bottom skin
# (~0.7 g/cm^3 of 27.1 cm^3). Weigh them and replace these.
SLED_MASS = 0.083
SLED_FLOOR_MASS = 0.019
# Sled print-frame geometry (mm), measured from sled.stl: inner closed-end face,
# and the floor's z span (it rests on the 2.5 mm lips that start at z ~ 33.3).
SLED_INNER_FRONT_Y = -76.55
SLED_FLOOR_Z = (30.8, 33.3)
# UGREEN 145 W 25000 mAh (90597A): 160 x 80 x 26 mm. It lies on the floor, its
# front against the closed end, so ~4.5 mm sticks out of the open end.
BANK_SIZE = np.array([80.0, 160.0, 26.0])  # Sled x, y, z.

SO101_MESHES = (
  "base_motor_holder_so101_v1",
  "sts3215_03a_v1",
  "sts3215_03a_no_horn_v1",
  "waveshare_mounting_plate_so101_v2",
  "motor_holder_so101_base_v1",
  "rotation_pitch_so101_v1",
  "upper_arm_so101_v1",
  "under_arm_so101_v1",
  "motor_holder_so101_wrist_v1",
  "wrist_roll_pitch_so101_v2",
  "moving_jaw_so101_v1",
)
SO101_MAX_FACES = 12_000

# Wrist camera module on the mount's plate, in the mount's print frame (mm):
# plate centroid and its normal (pointing away from the fingers), measured from
# Wrist_Cam_Mount_32x32_UVC_Module_SO101.stl. The board sits on the plate's
# finger side; the module is a 32 x 32 x 4 mm board + a 14 mm lens holder 10 mm
# long. The camera looks along -normal, down the fingers.
CAM_PLATE_CENTER = np.array([2.4, 69.3, -5.9])
CAM_PLATE_NORMAL = np.array([0.0, 0.368, -0.930])
CAM_PLATE_FACE = -8.0  # Finger-side face, along the normal from the centroid.
CAM_BOARD = (32.0, 32.0, 4.0)
CAM_LENS = (7.0, 10.0)  # Radius, length.


def camera_module() -> trimesh.Trimesh:
  """Board + lens holder of a 32 x 32 mm UVC module, in mount print frame (mm)."""
  n = CAM_PLATE_NORMAL / np.linalg.norm(CAM_PLATE_NORMAL)
  x = np.array([1.0, 0.0, 0.0])
  y = np.cross(n, x)
  frame = np.eye(4)
  frame[:3, :3] = np.c_[x, y, n]
  board = trimesh.creation.box(CAM_BOARD)
  board.apply_translation([0, 0, CAM_PLATE_FACE - CAM_BOARD[2] / 2])
  lens = trimesh.creation.cylinder(radius=CAM_LENS[0], height=CAM_LENS[1], sections=24)
  lens.apply_translation([0, 0, CAM_PLATE_FACE - CAM_BOARD[2] - CAM_LENS[1] / 2])
  module = trimesh.util.concatenate([board, lens])
  frame[:3, 3] = CAM_PLATE_CENTER
  module.apply_transform(frame)
  return module


def step_to_base(points_mm: np.ndarray) -> np.ndarray:
  """Map points from the Pupper STEP frame (mm) to the base_link frame (m)."""
  return (points_mm / 1000.0 - BASE_ORIGIN_IN_STEP) @ STEP_TO_BASE_ROT.T


def saddle_print_to_step(points: np.ndarray, saddle_v3: trimesh.Trimesh) -> np.ndarray:
  """Undo pupper_so101_mount.py's print export, with v3's bounding-box offset."""
  rotated = np.asarray(saddle_v3.vertices) @ SADDLE_PRINT_ROT.T
  lo, hi = rotated.min(axis=0), rotated.max(axis=0)
  offset = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
  return (points - offset) @ SADDLE_PRINT_ROT


def sled_holes(sled: trimesh.Trimesh) -> np.ndarray:
  """Centers (x, y) of the pin holes through the sled's base, print frame."""
  section = sled.section(plane_origin=[0, 0, 1.0], plane_normal=[0, 0, 1])
  assert section is not None
  planar, to_3d = section.to_2D()
  centers = []
  for polygon in planar.polygons_full:
    for hole in polygon.interiors:
      c = np.asarray(hole.coords).mean(axis=0)
      centers.append(trimesh.transform_points([[c[0], c[1], 0.0]], to_3d)[0][:2])
  assert len(centers) == 4, f"expected 4 pin holes, found {len(centers)}"
  return np.array(sorted(centers, key=lambda c: (c[1], c[0])))


def sled_to_step(sled: trimesh.Trimesh):
  """Map from sled print frame (mm) to STEP (mm), fit to the pin holes."""
  holes = sled_holes(sled)
  target = BELLY_PIN_HOLES[np.lexsort((BELLY_PIN_HOLES[:, 0], BELLY_PIN_HOLES[:, 1]))]
  # X = x + dx, Z = y + dz (least squares over the 4 holes), Y = belly - z.
  dx = np.mean(target[:, 0] - holes[:, 0])
  dz = np.mean(target[:, 1] - holes[:, 1])
  residual = np.c_[holes[:, 0] + dx - target[:, 0], holes[:, 1] + dz - target[:, 1]]
  print(f"sled pin holes vs belly holes: max error {np.abs(residual).max():.2f} mm")
  offset = np.array([dx, BELLY_UNDERSIDE_Y, dz])
  return lambda p: np.asarray(p) @ SLED_TO_STEP_ROT.T + offset


def load_mesh(path: Path) -> trimesh.Trimesh:
  mesh = trimesh.load(path, force="mesh")
  assert isinstance(mesh, trimesh.Trimesh), path
  return mesh


def mount_inertial(mesh: trimesh.Trimesh, mass: float) -> str:
  """MJCF <inertial> for a printed part of uniform density in base_link frame.

  The exported mounts are not perfectly watertight, so the inertia is computed
  here (trimesh tolerates small gaps) instead of by MuJoCo from the mesh.
  """
  mesh = mesh.copy()
  mesh.density = mass / mesh.volume
  inertia = mesh.moment_inertia  # About the center of mass.
  com = mesh.center_mass
  full = (inertia[0, 0], inertia[1, 1], inertia[2, 2])
  full += (inertia[0, 1], inertia[0, 2], inertia[1, 2])
  return (
    f'<inertial pos="{com[0]:.5f} {com[1]:.5f} {com[2]:.5f}" mass="{mass}" '
    f'fullinertia="{" ".join(f"{v:.4g}" for v in full)}"/>'
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Regenerate the Pupper v3 + SO-101 mesh assets."
  )
  parser.add_argument("--mount-dir", type=Path, required=True)
  parser.add_argument(
    "--so101-dir", type=Path, help="Omit to keep the SO-101 meshes in the repo."
  )
  parser.add_argument("--saddle-print", type=Path)
  parser.add_argument("--wrist-cam-mount", type=Path)
  parser.add_argument("--sled", type=Path, required=True)
  parser.add_argument("--sled-floor", type=Path, required=True)
  args = parser.parse_args()
  ASSET_DIR.mkdir(parents=True, exist_ok=True)

  for src, dst, mass in MOUNTS:
    mesh = load_mesh(args.mount_dir / src)
    if args.saddle_print and dst == "pupper_so101_saddle.stl":
      saddle_v3 = mesh
      mesh = load_mesh(args.saddle_print)
      mesh.vertices = saddle_print_to_step(np.asarray(mesh.vertices), saddle_v3)
      mass = round(mass * mesh.volume / SADDLE_V3_VOLUME_MM3, 3)
    mesh.vertices = step_to_base(np.asarray(mesh.vertices))
    mesh.export(ASSET_DIR / dst)
    print(f"{dst}: {len(mesh.faces)} faces, bounds {mesh.bounds.round(4).tolist()}")
    print(f"  paste into the XML: {mount_inertial(mesh, mass)}")

  sled = load_mesh(args.sled)
  to_step = sled_to_step(sled)
  floor = load_mesh(args.sled_floor)
  fv = np.asarray(floor.vertices)
  # Floor: centered across, front against the closed end, on the lips.
  shift = np.array(
    [
      -(fv[:, 0].min() + fv[:, 0].max()) / 2,
      SLED_INNER_FRONT_Y - fv[:, 1].min(),
      SLED_FLOOR_Z[0] - fv[:, 2].min(),
    ]
  )
  for mesh, dst, mass in (
    (sled, "pupper_so101_sled.stl", SLED_MASS),
    (floor, "pupper_so101_sled_floor.stl", SLED_FLOOR_MASS),
  ):
    pts = np.asarray(mesh.vertices) + (shift if mesh is floor else 0.0)
    mesh.vertices = step_to_base(to_step(pts))
    mesh.export(ASSET_DIR / dst)
    print(f"{dst}: bounds {mesh.bounds.round(4).tolist()}")
    print(f"  paste into the XML: {mount_inertial(mesh, mass)}")
  bank_lo = np.array(
    [-BANK_SIZE[0] / 2, SLED_INNER_FRONT_Y, SLED_FLOOR_Z[0] - BANK_SIZE[2]]
  )
  corners = step_to_base(to_step(np.array([bank_lo, bank_lo + BANK_SIZE])))
  center, half = corners.mean(axis=0), np.abs(np.diff(corners, axis=0))[0] / 2
  print(
    f'power bank box: pos="{" ".join(f"{v:.5f}" for v in center)}" '
    f'size="{" ".join(f"{v:.5f}" for v in half)}"'
  )

  if args.wrist_cam_mount:
    mount = load_mesh(args.wrist_cam_mount)
    if len(mount.faces) > SO101_MAX_FACES:
      mount = mount.simplify_quadric_decimation(face_count=SO101_MAX_FACES)
    for mesh, dst in (
      (mount, "so101_wrist_roll_follower_cam_mount.stl"),
      (camera_module(), "so101_wrist_camera_module.stl"),
    ):
      mesh.apply_scale(0.001)
      mesh.export(ASSET_DIR / dst)
      print(f"{dst}: {len(mesh.faces)} faces, bounds {mesh.bounds.round(4).tolist()}")

  for name in SO101_MESHES if args.so101_dir else ():
    src = args.so101_dir / f"{name}.stl"
    dst = ASSET_DIR / f"so101_{name}.stl"
    mesh = load_mesh(src)
    if len(mesh.faces) > SO101_MAX_FACES:
      mesh = mesh.simplify_quadric_decimation(face_count=SO101_MAX_FACES)
      mesh.export(dst)
    else:
      shutil.copyfile(src, dst)
    print(f"so101_{name}.stl: {len(mesh.faces)} faces")


if __name__ == "__main__":
  main()
