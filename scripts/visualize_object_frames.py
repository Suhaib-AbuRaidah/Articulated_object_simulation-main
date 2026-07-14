#!/usr/bin/env python3
"""Load an articulated object in PyBullet and visualize its coordinate frames.

Point this at a URDF from ``data/urdfs/Dataset/...``. The URDFs live inside
``.tar.gz`` archives (each holding a ``model.urdf``), so the ``path`` argument
accepts any of:

  * a ``.tar.gz`` archive -- it is extracted to a temp dir and its ``model.urdf``
    is loaded;
  * a directory anywhere under ``data/urdfs/Dataset`` -- the first archive found
    beneath it is loaded (use ``--list`` to see all of them, ``--index N`` to
    pick one);
  * a plain ``.urdf`` file;
  * a ``.npz`` sample whose ``object_path`` field points at a URDF.

Every joint starts at its zero state, and RGB coordinate frames are drawn for:

  * the object (base) frame,
  * every joint frame (at the joint origin, with its axis in yellow),
  * every link/part frame.

In the GUI a **slider is added for each movable joint** (ranged over its limits);
dragging a slider moves that joint and the link/joint frames follow live.

Axis colouring follows the usual convention: X = red, Y = green, Z = blue.

Usage
-----
    # Load the first lamp archive under a category
    python scripts/visualize_object_frames.py \
        data/urdfs/Dataset/articulated_task_lamp

    # List every archive under a directory
    python scripts/visualize_object_frames.py data/urdfs/Dataset --list

    # Load a specific archive directly
    python scripts/visualize_object_frames.py \
        data/urdfs/Dataset/.../rec_articulated_task_lamp_XXXX.tar.gz
"""

import argparse
import atexit
import os
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

# Temp dirs created while extracting archives; cleaned up at exit.
_TEMP_DIRS = []


def _cleanup_temp_dirs():
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def _find_archives(directory):
    """Return a sorted list of ``.tar.gz`` archives found under ``directory``."""
    matches = []
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".tar.gz"):
                matches.append(os.path.join(root, name))
    return sorted(matches)


def _extract_urdf_from_archive(archive_path):
    """Extract ``archive_path`` to a temp dir and return its ``model.urdf``."""
    tmp = tempfile.mkdtemp(prefix="urdf_viz_")
    _TEMP_DIRS.append(tmp)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(tmp)
    urdfs = []
    for root, _dirs, files in os.walk(tmp):
        for name in files:
            if name.endswith(".urdf"):
                urdfs.append(os.path.join(root, name))
    if not urdfs:
        sys.exit(f"No .urdf found inside archive: {archive_path}")
    # Prefer a file literally named model.urdf if present.
    for u in urdfs:
        if os.path.basename(u) == "model.urdf":
            return u
    return sorted(urdfs)[0]

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - dependency hint
    sys.exit(
        "pybullet is required to run this script. Install it with "
        "`pip install pybullet`.\n"
        f"(import error: {exc})"
    )


def resolve_urdf_path(path: str, index: int = 0, list_only: bool = False) -> str:
    """Resolve ``path`` to a loadable ``.urdf`` file.

    Handles ``.tar.gz`` archives, directories (searched for archives), plain
    ``.urdf`` files, and ``.npz`` samples (via their ``object_path`` field).
    """
    path = os.path.expanduser(path)

    if os.path.isdir(path):
        archives = _find_archives(path)
        if not archives:
            sys.exit(f"No .tar.gz archives found under: {path}")
        if list_only:
            print(f"{len(archives)} archive(s) under '{path}':")
            for i, a in enumerate(archives):
                print(f"  [{i}] {os.path.relpath(a, path)}")
            sys.exit(0)
        if not 0 <= index < len(archives):
            sys.exit(f"--index {index} out of range (0..{len(archives) - 1}).")
        archive = archives[index]
        print(f"Directory '{path}' -> archive [{index}]: {archive}")
        return _extract_urdf_from_archive(archive)

    if path.endswith(".tar.gz"):
        print(f"Archive: {path}")
        return _extract_urdf_from_archive(path)

    if path.lower().endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "object_path" not in data.files:
            sys.exit(f"'{path}' has no 'object_path' field; cannot locate a URDF.")
        urdf_path = str(data["object_path"])
        print(f"Sample '{path}' -> object_path: {urdf_path}")
        if not os.path.isfile(urdf_path):
            sys.exit(f"URDF not found: {urdf_path}")
        return urdf_path

    if not os.path.isfile(path):
        sys.exit(f"URDF not found: {path}")
    return path


def draw_frame(position, orientation, label, axis_len=0.15, line_width=3.0,
               ids=None):
    """Draw (or redraw) an RGB coordinate frame at (position, quaternion).

    When ``ids`` (from a previous call with the same ``label`` setting) is given,
    the existing debug items are replaced in place instead of accumulating new
    ones -- this lets the frame follow a moving link without flicker.  Returns
    the list of debug-item ids (3 axis lines, plus 1 text item if ``label``).
    """
    rot = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    origin = np.asarray(position, dtype=float)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # X=red, Y=green, Z=blue
    new_ids = []
    for axis in range(3):
        end = origin + rot[:, axis] * axis_len
        kwargs = dict(lineWidth=line_width, lifeTime=0)
        if ids is not None and ids[axis] >= 0:
            kwargs["replaceItemUniqueId"] = ids[axis]
        new_ids.append(p.addUserDebugLine(
            origin.tolist(), end.tolist(), colors[axis], **kwargs))
    if label:
        kwargs = dict(textColorRGB=(1, 1, 1), textSize=1.0)
        if ids is not None and len(ids) > 3 and ids[3] >= 0:
            kwargs["replaceItemUniqueId"] = ids[3]
        new_ids.append(p.addUserDebugText(
            label, (origin + 0.02).tolist(), **kwargs))
    return new_ids


def draw_dynamic_frames(body, num_joints, axis_len, no_labels, registry):
    """Draw/redraw every link frame and joint axis at the current joint state.

    ``registry`` holds the debug-item ids between calls so items are replaced
    (not duplicated) as the joints move.  Call once to create, then again after
    each joint change.
    """
    for j in range(num_joints):
        info = p.getJointInfo(body, j)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        link_name = info[12].decode("utf-8")

        link_state = p.getLinkState(body, j, computeForwardKinematics=True)
        link_pos = link_state[4]   # worldLinkFramePosition
        link_orn = link_state[5]   # worldLinkFrameOrientation

        # Link / part frame.
        part_label = None if no_labels else f"part:{link_name}"
        registry["frame"][j] = draw_frame(
            link_pos, link_orn, part_label, axis_len=axis_len,
            ids=registry["frame"].get(j))

        # Joint axis (yellow) for movable joints.
        if joint_type != p.JOINT_FIXED:
            rot = np.array(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
            axis_local = np.asarray(info[13], dtype=float)  # axis in child frame
            axis_world = rot @ axis_local
            origin = np.asarray(link_pos, dtype=float)
            start = origin - axis_world * axis_len
            end = origin + axis_world * axis_len
            kwargs = dict(lineWidth=4.0, lifeTime=0)
            if registry["axis"].get(j, -1) >= 0:
                kwargs["replaceItemUniqueId"] = registry["axis"][j]
            registry["axis"][j] = p.addUserDebugLine(
                start.tolist(), end.tolist(), (1, 1, 0), **kwargs)
            if not no_labels:
                kwargs = dict(textColorRGB=(1, 1, 0), textSize=1.0)
                if registry["axistxt"].get(j, -1) >= 0:
                    kwargs["replaceItemUniqueId"] = registry["axistxt"][j]
                registry["axistxt"][j] = p.addUserDebugText(
                    f"joint:{joint_name}", end.tolist(), **kwargs)


def build_joint_sliders(body, num_joints):
    """Add one GUI slider per movable joint. Returns [(joint_index, param_id)]."""
    sliders = []
    for j in range(num_joints):
        info = p.getJointInfo(body, j)
        joint_type = info[2]
        if joint_type == p.JOINT_FIXED:
            continue
        joint_name = info[1].decode("utf-8")
        lower, upper = info[8], info[9]
        if upper <= lower:  # unlimited (e.g. continuous) -> pick a usable range
            lower, upper = (-0.5, 0.5) if joint_type == p.JOINT_PRISMATIC \
                else (-np.pi, np.pi)
        start = float(min(max(0.0, lower), upper))  # start at 0 (clamped)
        param = p.addUserDebugParameter(f"{j}:{joint_name}", lower, upper, start)
        sliders.append((j, param))
    return sliders


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path",
                        help="A .tar.gz archive, a directory under "
                             "data/urdfs/Dataset, a .urdf file, or a .npz sample.")
    parser.add_argument("--index", type=int, default=0,
                        help="When 'path' is a directory, which archive to load.")
    parser.add_argument("--list", action="store_true",
                        help="List archives under a directory and exit.")
    parser.add_argument("--axis-len", type=float, default=0.15,
                        help="Length of the drawn coordinate axes (metres).")
    parser.add_argument("--no-labels", action="store_true",
                        help="Do not draw text labels next to the frames.")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the GUI (prints frames and exits).")
    args = parser.parse_args()

    urdf_path = resolve_urdf_path(args.path, index=args.index, list_only=args.list)

    mode = p.DIRECT if args.headless else p.GUI
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # static visualization, no dynamics

    if mode == p.GUI:
        # Keep the GUI panel ON -- it hosts the joint sliders. Only turn off the
        # camera-preview thumbnails to reduce clutter.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    body = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=client)

    num_joints = p.getNumJoints(body)

    # --- Set every joint to its zero state ------------------------------------
    for j in range(num_joints):
        j_info = p.getJointInfo(body, j)  # ensure joint info is cached
        joint_limits = (j_info[8], j_info[9])  # lower, upper
        print(f"Joint limits for joint {j} ({j_info[1].decode('utf-8')}): {joint_limits}")    
        p.resetJointState(body, j, targetValue=0, targetVelocity=0.0)

    # --- Object (base) frame (static; base is fixed) --------------------------
    base_pos, base_orn = p.getBasePositionAndOrientation(body)
    label = None if args.no_labels else "object (base)"
    draw_frame(base_pos, base_orn, label, axis_len=args.axis_len * 1.4,
               line_width=5.0)
    print("\nObject (base) frame:")
    print(f"  position    = {np.round(base_pos, 4).tolist()}")
    print(f"  orientation = {np.round(base_orn, 4).tolist()} (quat xyzw)")

    # --- Log joints/links -----------------------------------------------------
    type_names = {
        p.JOINT_REVOLUTE: "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_FIXED: "fixed",
        p.JOINT_PLANAR: "planar",
        p.JOINT_SPHERICAL: "spherical",
    }
    print(f"\n{num_joints} joint(s) / link(s):")
    for j in range(num_joints):
        info = p.getJointInfo(body, j)
        jtype = type_names.get(info[2], str(info[2]))
        link_pos = p.getLinkState(body, j, computeForwardKinematics=True)[4]
        print(f"  [{j}] joint='{info[1].decode()}' ({jtype}) -> "
              f"link='{info[12].decode()}'  frame pos = {np.round(link_pos, 4).tolist()}")

    print("\nLegend: X=red  Y=green  Z=blue  |  joint axis=yellow")

    # Registry of debug-item ids so the frames can be redrawn in place.
    registry = {"frame": {}, "axis": {}, "axistxt": {}}
    draw_dynamic_frames(body, num_joints, args.axis_len, args.no_labels, registry)

    if mode == p.GUI:
        sliders = build_joint_sliders(body, num_joints)
        print(f"\n{len(sliders)} joint slider(s) added -- drag them to move the "
              "joints; frames follow.")
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
            cameraTargetPosition=base_pos,
        )
        print("Close the window or press Ctrl+C to exit.")
        last = {j: None for j, _ in sliders}
        try:
            while p.isConnected():
                changed = False
                for j, param in sliders:
                    value = p.readUserDebugParameter(param)
                    if last[j] is None or abs(value - last[j]) > 1e-6:
                        p.resetJointState(body, j, targetValue=value,
                                          targetVelocity=0.0)
                        last[j] = value
                        changed = True
                if changed:  # redraw frames only when a joint actually moved
                    draw_dynamic_frames(body, num_joints, args.axis_len,
                                        args.no_labels, registry)
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            pass

    p.disconnect()


if __name__ == "__main__":
    main()
