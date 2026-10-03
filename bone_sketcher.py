"""Bone Sketcher: sketch a joint chain on a plane fitted to the model.

1. Fit Plane: fit an oriented sketch plane to the selected object/components.
2. Draw Points: click/drag points on the plane (native draggerContext).
3. Create Joints: build a chain through the points; every joint aims at the
   next one and shares the same up vector, so the chain has one rotation plane.

The scene is the source of truth (nothing is cached in Python), which keeps
Maya's undo consistent.

Joints are named [L_|R_]<name>_<n>, e.g. L_arm_1, L_arm_2 or spine_1.

Usage::

    import bone_sketcher
    bone_sketcher.ui()
"""

from __future__ import annotations

import math
from functools import partial

import maya.cmds as mc
import maya.mel as mm
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui

# Scene nodes (one sketch per session)
PLANE_NAME = "boneSketcher_plane"
PLANE_MATERIAL = "boneSketcher_planeMtl"
PLANE_ROOT = "boneSketcher_root"  # main mover, pivot at the bottom of the plane
AIM_GROUP = "boneSketcher_aimGrp"
AIM_HANDLE = "boneSketcher_aimHandle"  # the plane's Y always aims at this
UP_HANDLE = "boneSketcher_upHandle"  # the plane's Z points towards this
POINTS_GROUP = "boneSketcher_points"
POINT_PREFIX = "boneSketcher_pt"
PREVIEW_CURVE = "boneSketcher_preview"
POINT_ATTR = "boneSketcherPoint"
CAMERA_GROUP = "boneSketcher_cameras"
STORED_SET = "boneSketcher_storedComponents"
JOINT_ATTR = "boneSketcherJoint"  # tags joints created by the tool
CAMERA_PREFIX = "boneSketcher_"
CONTEXT_NAME = "boneSketcherDrawCtx"
PLANE_WIDTH_NODE = "boneSketcher_planeHalfWidth"  # drives the plane's width-side CVs
GIZMO_SCALE_NODE = "boneSketcher_gizmoScale"  # scale matrix shared by every gizmo curve

# Root control attributes
PLANE_WIDTH_ATTR = "planeWidth"  # multiplies the plane's fitted width
GIZMO_SCALE_ATTR = "gizmoScale"  # scales handle rings and axis arrows about their own origin

# Orthographic views around the drawing surface, in (width, Y, normal) terms:
# view -> (direction from plane centre to camera, camera up)
CAMERA_VIEWS = (
    ("top", (0, 0, 1), (0, 1, 0)),
    ("bottom", (0, 0, -1), (0, 1, 0)),
    ("front", (1, 0, 0), (0, 0, 1)),
    ("back", (-1, 0, 0), (0, 0, 1)),
    ("side", (0, 1, 0), (0, 0, 1)),
)
CAMERA_DISTANCE_RATIO = 3.0  # camera distance vs longest plane side
CAMERA_WIDTH_RATIO = 1.3  # orthographic width vs longest plane side

# Plane frame: local X / Y lie in the plane, local Z is the normal.
PLANE_MARGIN = 1.15  # plane is this much larger than the fitted extents
MIN_SIZE_RATIO = 0.1  # shortest plane side vs longest, for very thin selections
AXIS_COLORS = {
    "x": (1.0, 0.15, 0.15),
    "y": (0.15, 1.0, 0.15),
    "z": (0.25, 0.45, 1.0),
}
PLANE_COLOR = (0.35, 0.6, 1.0)
PLANE_TRANSPARENCY = 0.75
POINT_COLOR = (1.0, 0.85, 0.1)
PREVIEW_COLOR = (1.0, 0.55, 0.1)
POINT_SIZE_RATIO = 0.015  # point joint radius vs longest plane side
HANDLE_SIZE_RATIO = 0.04  # aim handle radius vs longest plane side
UP_HANDLE_DISTANCE_RATIO = 0.5  # up handle distance from the root vs plane height
ROOT_COLOR = (1.0, 1.0, 1.0)
PICK_RADIUS_PX = 12  # screen-space radius for clicking an existing point

# Joint creation options
AXIS_CHOICES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
# "Plane Normal" is always perpendicular to the visible plane; "Plane X" / "Plane Z"
# follow the red / blue axis arrows (which trade places when the normal is swapped).
UP_VECTOR_CHOICES = ("Plane Normal", "Plane X", "Plane Z")
SIDE_CHOICES = {"None": "", "Left": "L", "Right": "R"}  # UI label -> name prefix
DEFAULT_AIM_AXIS = "+Y"
DEFAULT_UP_AXIS = "+X"
DEFAULT_UP_VECTOR = "Plane Normal"
DEFAULT_JOINT_SIDE = "None"
DEFAULT_JOINT_NAME = "joint"

OPTION_PREFIX = "boneSketcherTool_"


# ---------------------------------------------------------------------------
# Plane fitting: principal axes of a point cloud


class Frame(object):
    """Oriented, right-handed frame fit to a point cloud.

    origin  -- centre of the fitted box
    x_axis  -- in-plane axis (second-longest spread)
    y_axis  -- longest spread of the points
    z_axis  -- plane normal (shortest spread); x_axis ^ y_axis == z_axis
    size    -- (extent along x, extent along y, extent along z)
    """

    def __init__(self, origin, x_axis, y_axis, z_axis, size):
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.size = size

    def matrix(self) -> list[float]:
        """Row-major 4x4 world matrix, as mc.xform(matrix=...) expects."""
        return (
            list(mvec_to_tuple(self.x_axis))
            + [0.0]
            + list(mvec_to_tuple(self.y_axis))
            + [0.0]
            + list(mvec_to_tuple(self.z_axis))
            + [0.0]
            + list(mvec_to_tuple(self.origin))
            + [1.0]
        )


def fit_frame(positions, y_hint=None, z_hint=None) -> Frame:
    """Fit a Frame to positions.

    y_hint -- optional direction the Y axis should point along (sign only)
    z_hint -- optional direction the normal should point along (sign only)
    """
    positions = [om.MVector(*p) for p in positions]
    if len(positions) < 2:
        raise ValueError("Need at least two distinct points to fit a plane.")

    centroid = om.MVector()
    for p in positions:
        centroid += p
    centroid = centroid / len(positions)
    offsets = [p - centroid for p in positions]

    cov = [[0.0] * 3 for _ in range(3)]
    for v in offsets:
        for i in range(3):
            for j in range(3):
                cov[i][j] += v[i] * v[j]

    eigenvalues, eigenvectors = jacobi_eigen_3x3(cov)
    order = sorted(range(3), key=lambda i: eigenvalues[i], reverse=True)
    y_axis = _normalized(om.MVector(*eigenvectors[order[0]]))
    z_axis = _normalized(om.MVector(*eigenvectors[order[2]]))

    if y_hint is not None and y_axis * om.MVector(*y_hint) < 0.0:
        y_axis = -y_axis
    if z_hint is not None and z_axis * om.MVector(*z_hint) < 0.0:
        z_axis = -z_axis

    # Re-derive X and Z so the frame is exactly orthonormal and right-handed.
    x_axis = _normalized(y_axis ^ z_axis)
    z_axis = _normalized(x_axis ^ y_axis)

    axes = (x_axis, y_axis, z_axis)
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for v in offsets:
        for i, axis in enumerate(axes):
            d = v * axis
            mins[i] = min(mins[i], d)
            maxs[i] = max(maxs[i], d)

    origin = om.MVector(centroid)
    for i, axis in enumerate(axes):
        origin += axis * ((mins[i] + maxs[i]) * 0.5)
    size = tuple(maxs[i] - mins[i] for i in range(3))
    if size[1] <= 1e-9:
        raise ValueError("Selected points are coincident; cannot fit a plane.")

    return Frame(origin, x_axis, y_axis, z_axis, size)


def jacobi_eigen_3x3(matrix, iterations=50, tolerance=1e-12):
    """Cyclic Jacobi eigen solver for a symmetric 3x3 matrix.

    Returns (eigenvalues, eigenvectors), eigenvectors[i] matching eigenvalues[i].
    """
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]

    for _ in range(iterations):
        p, q, max_val = 0, 1, 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j
        if max_val < tolerance:
            break

        if abs(a[p][p] - a[q][q]) < 1e-15:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * a[p][q], a[p][p] - a[q][q])
        c, s = math.cos(theta), math.sin(theta)

        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app + s * s * aqq + 2.0 * s * c * apq
        a[q][q] = s * s * app + c * c * aqq - 2.0 * s * c * apq
        a[p][q] = a[q][p] = 0.0
        for i in range(3):
            if i != p and i != q:
                aip, aiq = a[i][p], a[i][q]
                a[i][p] = a[p][i] = c * aip + s * aiq
                a[i][q] = a[q][i] = -s * aip + c * aiq
        for i in range(3):
            vip, viq = v[i][p], v[i][q]
            v[i][p] = c * vip + s * viq
            v[i][q] = -s * vip + c * viq

    eigenvalues = [a[i][i] for i in range(3)]
    eigenvectors = [tuple(v[i][j] for i in range(3)) for j in range(3)]
    return eigenvalues, eigenvectors


def mvec_to_tuple(vec) -> tuple[float, float, float]:
    return (vec[0], vec[1], vec[2])


def joint_name(side: str, name: str, index: int) -> str:
    """[side_]name_index, e.g. L_arm_1 or spine_1."""
    parts = [side, name, str(index)] if side else [name, str(index)]
    return "_".join(parts)


def _normalized(vec: om.MVector) -> om.MVector:
    if vec.length() < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vec.normal()


# ---------------------------------------------------------------------------
# Sketch plane
#
# The plane is a single-patch NURBS surface (Y = longest side of the selection, Z = normal
# by default) driven by a small rig, all pivoting at the bottom of the Y side:
#
#   root            main mover; moves plane, aim and up handles together
#     aimHandle     the plane's Y aims at it
#     upHandle      the plane's Z points towards it (rolls the plane)
#     aimGrp        aim-constrained to aimHandle / upHandle
#       plane       free to rotate on top of the aim, for manual tweaks
#         points    sketch point joints
#         cameras   hidden orthographic sketch cameras
#
# Coloured axis curves are extra shapes under the plane transform.
#
# The root carries two display attributes, both live through node connections:
# planeWidth drives the plane's width-side CVs, and gizmoScale drives a shared
# scale matrix that every handle ring / axis arrow curve is passed through
# (transformGeometry from a hidden original shape), so each one scales about
# its own transform's origin.
#
# The frame (what joints use for their up vector) and the drawing surface (what
# points are clicked onto and the cameras look at) are tracked separately: the
# surface normal is stored in plane-local space, so "Swap Normal" can turn the
# frame's X/Z without moving the surface.

_EXTENTS_ATTR = "fitExtents"  # selection extents along the fitted X / Y / Z
_SWAPPED_ATTR = "normalSwapped"  # True while frame X / Z are swapped
_SURFACE_ATTR = "surfaceNormal"  # drawing-surface normal in plane-local space


def get_plane() -> str | None:
    return PLANE_NAME if mc.objExists(PLANE_NAME) else None


def require_plane() -> str:
    plane = get_plane()
    if not plane:
        raise RuntimeError("No sketch plane yet. Select an object or components and fit a plane first.")
    return plane


def plane_frame(plane=None):
    """Current world-space (origin, x_axis, y_axis, normal) of the plane, as MVectors.

    Always read from the scene so manual adjustments by the user are respected.
    """
    plane = plane or require_plane()
    m = om.MMatrix(mc.xform(plane, q=True, m=True, ws=True))
    x_axis = om.MVector(m[0], m[1], m[2]).normal()
    y_axis = om.MVector(m[4], m[5], m[6]).normal()
    normal = om.MVector(m[8], m[9], m[10]).normal()
    origin = om.MVector(m[12], m[13], m[14])
    return origin, x_axis, y_axis, normal


def surface_frame(plane=None):
    """World-space (origin, in-plane width axis, in-plane Y axis, surface normal) of the
    drawing surface, as MVectors. Differs from plane_frame() while X/Z are swapped."""
    plane = plane or require_plane()
    m = om.MMatrix(mc.xform(plane, q=True, m=True, ws=True))
    origin = om.MVector(m[12], m[13], m[14])
    return tuple([origin] + [(v * m).normal() for v in surface_axes_local(plane)])


def surface_axes_local(plane=None):
    """Plane-local (width axis, Y axis, surface normal); width ^ Y == normal."""
    plane = plane or require_plane()
    normal = om.MVector(0.0, 0.0, 1.0)
    if mc.attributeQuery(_SURFACE_ATTR, node=plane, exists=True):
        normal = om.MVector(*mc.getAttr(plane + "." + _SURFACE_ATTR)[0]).normal()
    y_axis = om.MVector(0.0, 1.0, 0.0)
    return (y_axis ^ normal).normal(), y_axis, normal


def plane_dimensions(plane=None) -> tuple[float, float]:
    """(width, height) of the visible plane in its own units."""
    plane = plane or require_plane()
    extent_x, extent_y, _ = mc.getAttr(plane + "." + _EXTENTS_ATTR)[0]
    return _plane_dimensions(extent_x, extent_y)


def plane_size(plane=None) -> float:
    """Longest world-space side of the plane (used to scale points/joints)."""
    plane = plane or require_plane()
    bbox = mc.exactWorldBoundingBox(plane + "|" + _plane_shape_name())
    return max(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]) or 1.0


def fit_plane_to_selection() -> str:
    """Fit (or refit) the sketch plane to the current selection.

    Accepts mesh objects and any mesh components (vertices, edges, faces, UVs...).
    Refitting an existing plane keeps its sketch points.
    """
    selection = mc.ls(sl=True)
    plane = get_plane()
    if plane:
        # Never fit the plane to itself or its own sketch points.
        selection = [s for s in selection if not belongs_to_plane(s, plane)]
    if not selection:
        raise RuntimeError("Select a mesh object or mesh components to fit the plane to.")

    vertices = mc.polyListComponentConversion(selection, toVertex=True) or []
    if not vertices:
        raise RuntimeError("Selection contains no polygon geometry.")
    positions = _vertex_world_positions(vertices)

    centroid = [sum(p[i] for p in positions) / len(positions) for i in range(3)]
    try:
        frame = fit_frame(positions, y_hint=_y_hint(selection, centroid), z_hint=_z_hint(centroid))
    except ValueError as exc:
        raise RuntimeError(str(exc))

    # The root's origin (= pivot of the whole rig) sits at the bottom of the plane's Y side.
    _, height = _plane_dimensions(frame.size[0], frame.size[1])
    matrix = frame.matrix()
    for i in range(3):
        matrix[12 + i] -= frame.y_axis[i] * height * 0.5

    plane = plane or _create_plane_rig()
    _ensure_plane_attrs(plane)
    point_positions = _sketch_point_world_positions(plane)
    mc.setAttr(plane + "." + _EXTENTS_ATTR, *frame.size)
    mc.setAttr(plane + "." + _SWAPPED_ATTR, False)
    mc.setAttr(plane + "." + _SURFACE_ATTR, 0.0, 0.0, 1.0)

    mc.xform(PLANE_ROOT, m=matrix, ws=True)
    mc.setAttr(AIM_HANDLE + ".translate", 0.0, height, 0.0)
    mc.setAttr(UP_HANDLE + ".translate", 0.0, 0.0, height * UP_HANDLE_DISTANCE_RATIO)
    mc.setAttr(plane + ".rotate", 0.0, 0.0, 0.0)
    _rebuild_handle_shapes(max(frame.size[0], height))

    _restore_sketch_point_world_positions(point_positions)
    _rebuild_plane(plane)
    mc.select(PLANE_ROOT, r=True)
    return plane


def rotate_plane_90() -> None:
    """Roll the whole rig (root, handles, plane and cameras) 90 degrees about the root's Y axis.

    Useful for round shapes where the fitted orientation is ambiguous.
    Already-placed points keep their world positions.
    """
    plane = require_plane()
    point_positions = _sketch_point_world_positions(plane)
    mc.rotate(0.0, 90.0, 0.0, PLANE_ROOT, r=True, os=True)
    _restore_sketch_point_world_positions(point_positions)


def flip_direction() -> None:
    """Swap the root and the tip (aim handle), so the plane's Y points the other way.

    The root turns 180 degrees about its own Z, so the plane normal is kept,
    and moves to the far end of the drawing surface; the aim handle keeps its
    distance, now pointing back (by default it lands where the root was). The up
    handle keeps its offset from the root, so the roll is kept. The drawing
    surface covers the same area and placed points stay where they are.
    """
    plane = require_plane()
    point_positions = _sketch_point_world_positions(plane)
    old_root = om.MVector(*mc.xform(PLANE_ROOT, q=True, t=True, ws=True))
    to_tip = om.MVector(*mc.xform(AIM_HANDLE, q=True, t=True, ws=True)) - old_root
    up_offset = om.MVector(*mc.xform(UP_HANDLE, q=True, t=True, ws=True)) - old_root
    new_root = old_root + to_tip.normal() * plane_dimensions(plane)[1]

    mc.rotate(0.0, 0.0, 180.0, PLANE_ROOT, r=True, os=True)
    mc.xform(PLANE_ROOT, t=mvec_to_tuple(new_root), ws=True)
    mc.xform(AIM_HANDLE, t=mvec_to_tuple(new_root - to_tip), ws=True)
    mc.xform(UP_HANDLE, t=mvec_to_tuple(new_root + up_offset), ws=True)
    _restore_sketch_point_world_positions(point_positions)


def swap_normal() -> bool:
    """Swap the frame's X and Z axes while the visible plane stays exactly where it is.

    The frame turns 90 degrees about Y (pressing again turns it back exactly)
    and the surface is rebuilt in the new local space, so the drawing surface,
    points and cameras do not move. While swapped, the frame's Z (the joints'
    "Plane Normal" up vector) lies in the drawing surface.
    Returns True while swapped.
    """
    plane = require_plane()
    _ensure_plane_attrs(plane)
    surface_normal_world = surface_frame(plane)[3]
    swapped = mc.getAttr(plane + "." + _SWAPPED_ATTR)
    # Turning -90 makes the new +X the old +Z, so the red arrow points out of the
    # front of the surface (same side the blue arrow did before the swap).
    _rotate_plane_about_y(90.0 if swapped else -90.0)
    swapped = not swapped

    matrix = om.MMatrix(mc.xform(plane, q=True, m=True, ws=True))
    local = surface_normal_world * matrix.inverse()
    # Always a +/- frame axis by construction; rounding removes float drift.
    mc.setAttr(plane + "." + _SURFACE_ATTR, *[float(round(c)) for c in (local.x, local.y, local.z)])
    mc.setAttr(plane + "." + _SWAPPED_ATTR, swapped)
    _rebuild_plane(plane)
    return swapped


def delete_sketch() -> None:
    scalers = []
    if mc.objExists(GIZMO_SCALE_NODE):
        scalers = mc.listConnections(GIZMO_SCALE_NODE + ".outputMatrix", s=False, d=True) or []
    nodes = scalers + [
        PLANE_ROOT,
        CAMERA_GROUP,
        STORED_SET,
        PLANE_MATERIAL,
        PLANE_MATERIAL + "SG",
        GIZMO_SCALE_NODE,
        PLANE_WIDTH_NODE,
        PLANE_WIDTH_NODE + "Neg",
    ]
    for node in nodes:
        if mc.objExists(node):
            mc.delete(node)


def belongs_to_plane(node: str, plane: str) -> bool:
    """True for anything in the plane rig (root, handles, plane, points)."""
    long_name = (mc.ls(node.split(".")[0], long=True) or [""])[0]
    root_long = (mc.ls(PLANE_ROOT, long=True) or [None])[0]
    if root_long and (long_name == root_long or long_name.startswith(root_long + "|")):
        return True
    plane_long = mc.ls(plane, long=True)[0]
    return long_name == plane_long or long_name.startswith(plane_long + "|")


def _plane_shape_name() -> str:
    return PLANE_NAME + "Shape"


def _create_plane_rig() -> str:
    """Build root > (aimHandle, upHandle, aimGrp > plane > points). Returns the plane."""
    root = mc.createNode("transform", n=PLANE_ROOT)
    aim_handle = mc.createNode("transform", n=AIM_HANDLE, p=root)
    up_handle = mc.createNode("transform", n=UP_HANDLE, p=root)
    aim_grp = mc.createNode("transform", n=AIM_GROUP, p=root)

    # Degree-1, single-patch NURBS plane: four CVs, laid out by _rebuild_plane().
    plane = mc.nurbsPlane(ax=(0, 0, 1), w=1, lr=1, d=1, u=1, v=1, ch=False, n=PLANE_NAME)[0]
    plane = mc.parent(plane, aim_grp, r=True)[0]
    shape = mc.listRelatives(plane, shapes=True, fullPath=True)[0]
    shape = mc.rename(shape, _plane_shape_name())
    mc.setAttr(shape + ".doubleSided", True)
    mc.setAttr(shape + ".opposite", False)
    _assign_plane_material(plane)
    _ensure_plane_attrs(plane)
    mc.group(em=True, n=POINTS_GROUP, p=plane)

    mc.aimConstraint(
        aim_handle,
        aim_grp,
        aimVector=(0, 1, 0),
        upVector=(0, 0, 1),
        worldUpType="object",
        worldUpObject=up_handle,
        mo=False,
    )

    # Root moves / rotates everything; handles only translate; plane only rotates.
    _lock_attrs(root, "s")
    _lock_attrs(aim_handle, "rs")
    _lock_attrs(up_handle, "rs")
    _lock_attrs(aim_grp, "trs")
    _lock_attrs(PLANE_NAME, "ts")
    _ensure_root_attrs()
    return PLANE_NAME


def _ensure_root_attrs() -> None:
    """Add the root's display attributes (also upgrades rigs made by older versions)."""
    for name in (PLANE_WIDTH_ATTR, GIZMO_SCALE_ATTR):
        if not mc.attributeQuery(name, node=PLANE_ROOT, exists=True):
            mc.addAttr(PLANE_ROOT, ln=name, at="double", dv=1.0, min=0.01, k=True)


def _lock_attrs(node: str, channels: str) -> None:
    for channel in channels:
        for axis in "xyz":
            mc.setAttr("{}.{}{}".format(node, channel, axis), l=True, k=False, cb=False)
    mc.setAttr(node + ".visibility", k=False, cb=False)


def _rebuild_handle_shapes(size: float) -> None:
    """(Re)draw the root / aim / up handle curves, scaled to the plane."""
    _ensure_root_attrs()
    radius = size * HANDLE_SIZE_RATIO
    specs = (
        # node, colour, ring normals, radius
        (PLANE_ROOT, ROOT_COLOR, ((0, 1, 0),), radius * 2.5),
        (AIM_HANDLE, AXIS_COLORS["y"], ((1, 0, 0), (0, 1, 0), (0, 0, 1)), radius),
        (UP_HANDLE, AXIS_COLORS["z"], ((1, 0, 0), (0, 1, 0), (0, 0, 1)), radius * 0.75),
    )
    for node, color, normals, ring_radius in specs:
        _delete_gizmo_curves(node)
        for index, normal in enumerate(normals):
            ring = mc.circle(nr=normal, r=ring_radius, ch=False)[0]
            _add_gizmo_curve(node, ring, color, "{}Shape{}".format(node, index + 1 if index else ""))


def _gizmo_scale_matrix() -> str:
    """Scale matrix driven by the root's gizmoScale, shared by every gizmo curve."""
    if not mc.objExists(GIZMO_SCALE_NODE):
        mc.createNode("composeMatrix", n=GIZMO_SCALE_NODE)
    source = PLANE_ROOT + "." + GIZMO_SCALE_ATTR
    for axis in "XYZ":
        target = GIZMO_SCALE_NODE + ".inputScale" + axis
        if not mc.isConnected(source, target):
            mc.connectAttr(source, target, f=True)
    return GIZMO_SCALE_NODE + ".outputMatrix"


def _add_gizmo_curve(node: str, temp_curve: str, color, name: str) -> str:
    """Move temp_curve's shape under node as a hidden original, displayed through a
    transformGeometry so the root's gizmoScale scales it about node's origin."""
    orig = mc.listRelatives(temp_curve, shapes=True, fullPath=True)[0]
    orig = mc.parent(orig, node, shape=True, r=True)[0]
    mc.delete(temp_curve)
    orig = mc.rename(orig, name + "Orig")
    mc.setAttr(orig + ".intermediateObject", True)

    shape = mc.createNode("nurbsCurve", n=name, p=node)
    scaler = mc.createNode("transformGeometry", n=name + "_scale")
    mc.connectAttr(orig + ".local", scaler + ".inputGeometry")
    mc.connectAttr(_gizmo_scale_matrix(), scaler + ".transform")
    mc.connectAttr(scaler + ".outputGeometry", shape + ".create")
    _set_color(shape, color)
    mc.setAttr(shape + ".lineWidth", 2.0)
    return shape


def _delete_gizmo_curves(node: str) -> None:
    """Delete node's curve shapes, their hidden originals and scale nodes."""
    nodes = []
    for shape in mc.listRelatives(node, shapes=True, type="nurbsCurve", fullPath=True) or []:
        nodes.append(shape)
        nodes += mc.listConnections(shape + ".create", s=True, d=False, type="transformGeometry") or []
    if nodes:
        mc.delete(nodes)


def _ensure_plane_attrs(plane: str) -> None:
    """Add the bookkeeping attributes (also upgrades planes made by older versions)."""
    if not mc.attributeQuery(_EXTENTS_ATTR, node=plane, exists=True):
        _add_double3(plane, _EXTENTS_ATTR, (1.0, 1.0, 1.0))
    if not mc.attributeQuery(_SURFACE_ATTR, node=plane, exists=True):
        _add_double3(plane, _SURFACE_ATTR, (0.0, 0.0, 1.0))
    if not mc.attributeQuery(_SWAPPED_ATTR, node=plane, exists=True):
        mc.addAttr(plane, ln=_SWAPPED_ATTR, at="bool")


def _add_double3(node: str, name: str, value) -> None:
    mc.addAttr(node, ln=name, at="double3")
    for axis in "XYZ":
        mc.addAttr(node, ln=name + axis, at="double", p=name)
    mc.setAttr(node + "." + name, *value)


def _assign_plane_material(plane: str) -> None:
    material = PLANE_MATERIAL
    shading_group = material + "SG"
    if not mc.objExists(material):
        mc.shadingNode("lambert", asShader=True, n=material)
        mc.setAttr(material + ".color", *PLANE_COLOR, type="double3")
        mc.setAttr(material + ".transparency", *([PLANE_TRANSPARENCY] * 3), type="double3")
    if not mc.objExists(shading_group):
        mc.sets(renderable=True, noSurfaceShader=True, empty=True, n=shading_group)
        mc.connectAttr(material + ".outColor", shading_group + ".surfaceShader", f=True)
    mc.sets(plane, e=True, forceElement=shading_group)


def _plane_dimensions(extent_x: float, extent_y: float) -> tuple[float, float]:
    width = extent_x * PLANE_MARGIN
    height = extent_y * PLANE_MARGIN
    width = max(width, height * MIN_SIZE_RATIO)
    return width, height


def _rebuild_plane(plane: str) -> None:
    """Lay out the surface and axis arrows for the current surface normal.

    The surface always spans the fitted X / Y extents, whichever frame axis is
    currently its normal, so swapping never changes the visible plane.
    """
    _ensure_root_attrs()
    width, height = plane_dimensions(plane)
    width_axis, y_axis, normal = surface_axes_local(plane)
    shape = plane + "|" + _plane_shape_name()
    half_width, neg_half_width = _plane_half_width_plugs(width * 0.5)
    # The width axis is always local +/-X or +/-Z; that CV component follows the root's planeWidth.
    width_index = 0 if abs(width_axis.x) > 0.5 else 2
    width_channel = "xValue" if width_index == 0 else "zValue"

    # U runs along the width axis and V along Y, so U ^ V (the surface normal) matches.
    # Local Y runs from 0 (bottom edge, the pivot) to height (tip, the aim handle).
    for u in range(2):
        for v in range(2):
            plug = "{}.controlPoints[{}]".format(shape, u * 2 + v)
            for channel in ("xValue", "yValue", "zValue"):
                for source in mc.listConnections(plug + "." + channel, s=True, d=False, p=True) or []:
                    mc.disconnectAttr(source, plug + "." + channel)
            mc.setAttr(plug, *mvec_to_tuple(y_axis * (v * height)))
            side = width_axis[width_index] * (u * 2 - 1)
            mc.connectAttr(half_width if side > 0 else neg_half_width, plug + "." + width_channel)
    _rebuild_axis_curves(plane, width, height, normal)


def _plane_half_width_plugs(half_width: float) -> tuple[str, str]:
    """(+, -) output plugs of fitted half width * the root's planeWidth."""
    positive = PLANE_WIDTH_NODE
    negative = PLANE_WIDTH_NODE + "Neg"
    if not mc.objExists(positive):
        mc.createNode("multDoubleLinear", n=positive)
    if not mc.objExists(negative):
        mc.createNode("multDoubleLinear", n=negative)
        mc.setAttr(negative + ".input2", -1.0)
    source = PLANE_ROOT + "." + PLANE_WIDTH_ATTR
    if not mc.isConnected(source, positive + ".input2"):
        mc.connectAttr(source, positive + ".input2", f=True)
    if not mc.isConnected(positive + ".output", negative + ".input1"):
        mc.connectAttr(positive + ".output", negative + ".input1", f=True)
    mc.setAttr(positive + ".input1", half_width)
    return positive + ".output", negative + ".output"


def _rebuild_axis_curves(plane: str, width: float, height: float, surface_normal: om.MVector) -> None:
    _delete_gizmo_curves(plane)

    longest = max(width, height)
    head = longest * 0.04

    def length(direction):
        # Whichever frame axis is the surface normal gets the short arrow.
        if abs(om.MVector(*direction) * surface_normal) > 0.5:
            return longest * 0.25
        return height if direction[1] else width * 0.5

    specs = {
        # axis: (direction, perpendicular used for the arrow head, length)
        "x": ((1, 0, 0), (0, 1, 0), length((1, 0, 0))),
        "y": ((0, 1, 0), (1, 0, 0), length((0, 1, 0))),
        "z": ((0, 0, 1), (0, 1, 0), length((0, 0, 1))),
    }
    for axis, (direction, perp, axis_length) in specs.items():
        tip = [d * axis_length for d in direction]
        back = [t - d * head * 2.0 for t, d in zip(tip, direction)]
        barb_a = [b + p * head for b, p in zip(back, perp)]
        barb_b = [b - p * head for b, p in zip(back, perp)]
        temp = mc.curve(d=1, p=[(0, 0, 0), tip, barb_a, tip, barb_b])
        _add_gizmo_curve(plane, temp, AXIS_COLORS[axis], "{}_{}AxisShape".format(PLANE_NAME, axis))


def _rotate_plane_about_y(angle: float) -> None:
    """Rotate the plane without moving already-placed sketch points."""
    plane = require_plane()
    point_positions = _sketch_point_world_positions(plane)
    mc.rotate(0.0, angle, 0.0, plane, r=True, os=True)
    _restore_sketch_point_world_positions(point_positions)


def _sketch_point_world_positions(plane: str) -> list:
    group = plane + "|" + POINTS_GROUP
    if not mc.objExists(group):
        return []
    return [
        (point, mc.xform(point, q=True, t=True, ws=True))
        for point in mc.listRelatives(group, children=True, type="transform", fullPath=True) or []
        if mc.attributeQuery(POINT_ATTR, node=point, exists=True)
    ]


def _restore_sketch_point_world_positions(point_positions: list) -> None:
    for point, position in point_positions:
        mc.xform(point, t=position, ws=True)


def _y_hint(selection: list[str], centroid) -> tuple[float, float, float]:
    """Make Y point away from the middle of the model the selection belongs to,
    falling back to away from the world origin, then world +Y."""
    objects = sorted({s.split(".")[0] for s in selection})
    bbox = mc.exactWorldBoundingBox(objects)
    model_centre = [(bbox[i] + bbox[i + 3]) * 0.5 for i in range(3)]
    diagonal = om.MVector(bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]).length()

    away = om.MVector(*centroid) - om.MVector(*model_centre)
    if away.length() > diagonal * 0.05:
        return mvec_to_tuple(away)
    away = om.MVector(*centroid)
    if away.length() > max(diagonal, 1e-6) * 0.05:
        return mvec_to_tuple(away)
    return (0.0, 1.0, 0.0)


def _z_hint(centroid) -> tuple[float, float, float] | None:
    """Make the normal face the camera of the active viewport, if there is one."""
    try:
        panel = mc.getPanel(withFocus=True)
        if not panel or mc.getPanel(typeOf=panel) != "modelPanel":
            panel = mc.playblast(activeEditor=True)
        camera = mc.modelPanel(panel, q=True, camera=True)
        position = mc.xform(camera, q=True, t=True, ws=True)
    except (RuntimeError, TypeError):
        return None
    return mvec_to_tuple(om.MVector(*position) - om.MVector(*centroid))


def _vertex_world_positions(vertices: list[str]) -> list[tuple[float, float, float]]:
    """World positions of vertex components, across any number of meshes.

    (mc.xform can only query one object at a time.)
    """
    selection = om.MSelectionList()
    for vertex in vertices:
        selection.add(vertex)
    positions = []
    iterator = om.MItSelectionList(selection, om.MFn.kMeshVertComponent)
    while not iterator.isDone():
        dag_path, component = iterator.getComponent()
        vertex_iter = om.MItMeshVertex(dag_path, component)
        while not vertex_iter.isDone():
            p = vertex_iter.position(om.MSpace.kWorld)
            positions.append((p.x, p.y, p.z))
            vertex_iter.next()
        iterator.next()
    return positions


# ---------------------------------------------------------------------------
# Sketch points
#
# Points are joints parented under the plane (so they follow manual plane
# adjustments), in click order. Joints rather than locators so they show
# through the model with the viewport's X-Ray Joints mode.


def points_group() -> str:
    plane = require_plane()
    group = plane + "|" + POINTS_GROUP
    if not mc.objExists(group):
        group = mc.group(em=True, n=POINTS_GROUP, p=plane)
    return group


def list_points() -> list[str]:
    plane = get_plane()
    if not plane or not mc.objExists(plane + "|" + POINTS_GROUP):
        return []
    children = mc.listRelatives(points_group(), children=True, type="transform", fullPath=True) or []
    return [c for c in children if is_point(c)]


def is_point(node: str) -> bool:
    return mc.objExists(node) and mc.attributeQuery(POINT_ATTR, node=node, exists=True)


def point_positions() -> list[om.MVector]:
    return [om.MVector(*mc.xform(p, q=True, t=True, ws=True)) for p in list_points()]


def intersect_plane(ray_source, ray_direction) -> om.MVector | None:
    """World-space hit of a ray with the (infinite) drawing surface, or None."""
    origin, _, _, normal = surface_frame()
    return _intersect(om.MVector(ray_source), om.MVector(ray_direction), origin, normal)


def add_point(world_position: om.MVector) -> str:
    # Created directly under the group with zero rotation, so the local axes match the plane.
    point = mc.createNode("joint", n=POINT_PREFIX + "1", p=points_group())
    point = mc.ls(point, long=True)[0]
    mc.addAttr(point, ln=POINT_ATTR, at="bool", dv=True)
    mc.xform(point, t=mvec_to_tuple(world_position), ws=True)

    mc.setAttr(point + ".radius", plane_size() * POINT_SIZE_RATIO)
    mc.setAttr(point + ".displayLocalAxis", True)
    _set_color(point, POINT_COLOR)
    for attr in ("rotate", "scale", "jointOrient"):
        for axis in "XYZ":
            mc.setAttr("{}.{}{}".format(point, attr, axis), l=True, k=False, cb=False)

    rebuild_preview()
    add_to_isolation([point])
    return point


def delete_point(point: str) -> None:
    if is_point(point):
        mc.delete(point)
    rebuild_preview()


def delete_last_point() -> None:
    points = list_points()
    if points:
        delete_point(points[-1])


def rebuild_preview() -> str | None:
    """Degree-1 curve through the points, its CVs driven live by the point translates."""
    group = points_group()
    existing = group + "|" + PREVIEW_CURVE
    if mc.objExists(existing):
        mc.delete(existing)

    points = list_points()
    if len(points) < 2:
        return None

    positions = [mc.getAttr(p + ".translate")[0] for p in points]
    curve = mc.curve(d=1, p=positions, n=PREVIEW_CURVE)
    curve = mc.parent(curve, group, r=True)[0]
    shape = mc.listRelatives(curve, shapes=True, fullPath=True)[0]
    for index, point in enumerate(points):
        mc.connectAttr(point + ".translate", "{}.controlPoints[{}]".format(shape, index))
    _set_color(shape, PREVIEW_COLOR)
    mc.setAttr(shape + ".lineWidth", 2.0)
    # Not pickable, so clicks always reach the points.
    mc.setAttr(shape + ".overrideDisplayType", 2)
    return curve


def _set_color(node: str, rgb) -> None:
    mc.setAttr(node + ".overrideEnabled", True)
    mc.setAttr(node + ".overrideRGBColors", True)
    mc.setAttr(node + ".overrideColorRGB", *rgb)


# ---------------------------------------------------------------------------
# Orthographic cameras around the sketch plane
#
# The cameras live under a hidden group parented to the plane, so they follow
# any roll / swap / aim / manual adjustment of it, centred on the middle of the
# plane. "top" looks straight down the
# plane normal -- the natural view for drawing points.

_previous_cameras = {}  # model panel -> camera it showed before a sketch view


def view_names() -> list[str]:
    return [view for view, _, _ in CAMERA_VIEWS]


def camera_name(view: str) -> str:
    return CAMERA_PREFIX + view


def update_cameras(plane=None) -> None:
    """Create the cameras if needed and fit them to the plane's current size."""
    plane = plane or require_plane()
    group = _camera_group(plane)
    longest = plane_size(plane)
    distance = longest * CAMERA_DISTANCE_RATIO

    surface_axes = surface_axes_local(plane)
    centre = surface_axes[1] * (plane_dimensions(plane)[1] * 0.5)
    for view, direction, up in CAMERA_VIEWS:
        name = camera_name(view)
        if not mc.objExists(name):
            transform, _ = mc.camera(orthographic=True)
            transform = mc.parent(transform, group, r=True)[0]
            mc.rename(transform, name)
        shape = mc.listRelatives(name, shapes=True, fullPath=True)[0]
        mc.setAttr(shape + ".orthographic", True)
        mc.setAttr(shape + ".orthographicWidth", longest * CAMERA_WIDTH_RATIO)
        mc.setAttr(shape + ".nearClipPlane", distance * 0.01)
        mc.setAttr(shape + ".farClipPlane", distance * 3.0)
        mc.xform(name, m=_camera_local_matrix(direction, up, distance, surface_axes, centre), os=True)


def look_through(view: str, panel=None) -> str:
    """Show a sketch camera (or "persp" to go back) in the given/active viewport."""
    panel = panel or active_model_panel()
    if not panel:
        raise RuntimeError("No 3D viewport found.")

    if view == "persp":
        camera = _previous_cameras.pop(panel, None)
        if not camera or not mc.objExists(camera):
            camera = "persp"
    else:
        require_plane()
        camera = camera_name(view)
        if not mc.objExists(camera):
            update_cameras()
        current = mc.modelPanel(panel, q=True, camera=True)
        if not _is_sketch_camera(current):
            _previous_cameras[panel] = current
    mc.lookThru(panel, camera)
    return camera


def restore_viewports() -> None:
    """Point every viewport that shows a sketch camera back at its previous camera."""
    for panel in mc.getPanel(type="modelPanel") or []:
        if _is_sketch_camera(mc.modelPanel(panel, q=True, camera=True)):
            look_through("persp", panel)


def active_model_panel() -> str | None:
    panel = mc.getPanel(withFocus=True)
    if panel and mc.getPanel(typeOf=panel) == "modelPanel":
        return panel
    try:
        return mc.playblast(activeEditor=True)
    except RuntimeError:
        panels = mc.getPanel(type="modelPanel") or []
        return panels[0] if panels else None


def _camera_group(plane: str) -> str:
    group = plane + "|" + CAMERA_GROUP
    if mc.objExists(group):
        return group
    if mc.objExists(CAMERA_GROUP):
        # Older sketches kept the group in the world, driven through offsetParentMatrix.
        for source in mc.listConnections(CAMERA_GROUP + ".offsetParentMatrix", s=True, d=False, p=True) or []:
            mc.disconnectAttr(source, CAMERA_GROUP + ".offsetParentMatrix")
        mc.setAttr(CAMERA_GROUP + ".offsetParentMatrix", om.MMatrix(), type="matrix")
        mc.parent(CAMERA_GROUP, plane, r=True)
        mc.xform(group, m=list(om.MMatrix()), os=True)
    else:
        mc.group(em=True, n=CAMERA_GROUP, p=plane)
        mc.setAttr(group + ".visibility", False)  # cameras still work when hidden
    return group


def _is_sketch_camera(camera: str | None) -> bool:
    if not camera:
        return False
    transform = camera
    if mc.objExists(camera) and mc.nodeType(camera) == "camera":
        transform = mc.listRelatives(camera, parent=True)[0]
    return transform.split("|")[-1] in {camera_name(v) for v in view_names()}


def _camera_local_matrix(direction, up, distance, surface_axes, centre) -> list[float]:
    """Camera matrix in the camera group's (plane-aligned) space.

    direction / up are given in drawing-surface terms (width, Y, normal) so the
    views stay put on the surface even while the frame's X / Z are swapped.
    """

    def to_local(v):
        return surface_axes[0] * v[0] + surface_axes[1] * v[1] + surface_axes[2] * v[2]

    # A camera looks down its -Z, so its Z axis points from the target to the camera.
    z_axis = to_local(direction).normal()
    y_axis = to_local(up).normal()
    x_axis = (y_axis ^ z_axis).normal()
    position = centre + z_axis * distance
    return (
        [x_axis.x, x_axis.y, x_axis.z, 0.0]
        + [y_axis.x, y_axis.y, y_axis.z, 0.0]
        + [z_axis.x, z_axis.y, z_axis.z, 0.0]
        + [position.x, position.y, position.z, 1.0]
    )


# ---------------------------------------------------------------------------
# Isolate the components the plane was fitted to, together with the sketch
#
# Fit Plane stores its selection in an objectSet, so it is saved with the scene
# and follows Maya's undo like everything else.


def store_selection(selection=None) -> list[str]:
    """Store the selected objects/components (sketch nodes are ignored). Returns the members."""
    selection = selection if selection is not None else mc.ls(sl=True)
    plane = get_plane()
    members = [s for s in selection if not (plane and belongs_to_plane(s, plane))]
    members = [s for s in members if not is_sketch_joint(s.split(".")[0])]
    if not members:
        raise RuntimeError("Select the objects or components to store.")
    if not mc.polyListComponentConversion(members, toFace=True):
        raise RuntimeError("Selection contains no polygon geometry to store.")

    if mc.objExists(STORED_SET):
        mc.sets(clear=STORED_SET)
        mc.sets(members, add=STORED_SET)
    else:
        mc.sets(members, n=STORED_SET)
    return stored_members()


def stored_members() -> list[str]:
    if not mc.objExists(STORED_SET):
        return []
    return mc.sets(STORED_SET, q=True) or []


def is_sketch_joint(node: str) -> bool:
    return mc.objExists(node) and mc.nodeType(node) == "joint" and mc.attributeQuery(JOINT_ATTR, node=node, exists=True)


def toggle_isolate(panel=None) -> bool:
    """Isolate the stored components + sketch + this tool's joints in the viewport, or undo it.

    Mirrors Maya's own Isolate Select (Ctrl+1): enable from the selection with
    enableIsolateSelect, then add every sketch node explicitly as a DAG object --
    when faces are part of an isolation, whole objects in the same selection
    are otherwise not reliably shown.
    Returns True when isolation is now on.
    """
    panel = panel or active_model_panel()
    if not panel:
        raise RuntimeError("No 3D viewport found.")
    _ensure_isolate_procs()
    if mc.isolateSelect(panel, q=True, state=True):
        mm.eval('enableIsolateSelect "{}" false;'.format(panel))
        return False

    components, dag_objects = _isolation_nodes()
    if not components and not dag_objects:
        raise RuntimeError("Nothing to isolate. Fit a plane to a selection first.")

    previous = mc.ls(sl=True)
    mc.select(components + dag_objects, r=True)
    mm.eval('enableIsolateSelect "{}" true;'.format(panel))
    for node in dag_objects:
        mc.isolateSelect(panel, addDagObject=node)
    if previous:
        mc.select(previous, r=True)
    else:
        mc.select(cl=True)
    return True


def add_to_isolation(nodes: list[str]) -> None:
    """Show newly created nodes (and their shapes) in every viewport that is currently isolated."""
    panels = [p for p in mc.getPanel(type="modelPanel") or [] if mc.isolateSelect(p, q=True, state=True)]
    if not panels:
        return
    dag_objects = _with_descendants(nodes)
    for panel in panels:
        for node in dag_objects:
            mc.isolateSelect(panel, addDagObject=node)


def _isolation_nodes() -> tuple[list[str], list[str]]:
    """(components, dag objects) to show."""
    components, dag_objects = [], []
    for member in stored_members():
        if "." in member:
            # Vertices / edges can't be isolated on their own; show their faces.
            components += mc.polyListComponentConversion(member, toFace=True) or []
        else:
            dag_objects.append(member)

    if mc.objExists(PLANE_ROOT):
        dag_objects.append(PLANE_ROOT)
    dag_objects += [j for j in mc.ls(type="joint", long=True) or [] if is_sketch_joint(j)]
    return components, _with_descendants(dag_objects)


def _with_descendants(nodes: list[str]) -> list[str]:
    """Nodes plus all their DAG descendants (transforms and shapes), long names, no duplicates."""
    result = []
    for node in mc.ls(nodes, long=True) or []:
        result.append(node)
        result += mc.listRelatives(node, allDescendents=True, fullPath=True) or []
    seen = set()
    return [n for n in result if not (n in seen or seen.add(n))]


def _ensure_isolate_procs() -> None:
    if not mm.eval('exists "enableIsolateSelect"'):
        mm.eval('source "createModelPanelMenu.mel";')


# ---------------------------------------------------------------------------
# Point-drawing tool (native draggerContext)
#
#   Click the plane       -> add a point where the mouse ray hits the drawing surface
#   Click / drag a point  -> select it; dragging slides it parallel to the plane
#   Shift+drag a point    -> move it along the plane normal (off the plane)
#   Ctrl+click a point    -> delete it
#
# While the tool is active the viewports use X-Ray Joints, so the points stay
# visible through the model.

_DRAW_HELP = (
    "Click the plane to add a point. Drag a point to slide it, Shift+drag to lift it off the plane, "
    "Ctrl+click to delete."
)

_drag = {}  # state of the current press/drag
_xray_states = {}  # model panel -> jointXray before the tool started


def enter_draw_tool() -> None:
    require_plane()
    ctx = CONTEXT_NAME
    # Always rebuild so the context runs the currently loaded code.
    if mc.contextInfo(ctx, exists=True):
        if mc.currentCtx() == ctx:
            mc.setToolTo("selectSuperContext")
        mc.deleteUI(ctx, toolContext=True)
    mc.draggerContext(
        ctx,
        pressCommand=_on_press,
        dragCommand=_on_drag,
        releaseCommand=_on_release,
        initialize=_on_tool_enter,
        finalize=_on_tool_exit,
        space="screen",
        cursor="crossHair",
        undoMode="all",
        image1="kinJoint.png",
    )
    mc.select(cl=True)
    mc.setToolTo(ctx)


def leave_draw_tool() -> None:
    if is_drawing():
        mc.setToolTo("selectSuperContext")


def is_drawing() -> bool:
    return mc.currentCtx() == CONTEXT_NAME


def _on_tool_enter() -> None:
    mc.headsUpMessage(_DRAW_HELP, time=4.0)
    _xray_states.clear()
    for panel in mc.getPanel(type="modelPanel") or []:
        _xray_states[panel] = mc.modelEditor(panel, q=True, jointXray=True)
        mc.modelEditor(panel, e=True, jointXray=True)


def _on_tool_exit() -> None:
    for panel, state in _xray_states.items():
        if mc.modelEditor(panel, exists=True):
            mc.modelEditor(panel, e=True, jointXray=state)
    _xray_states.clear()
    _drag.clear()


def _on_press() -> None:
    _drag.clear()
    ctx = CONTEXT_NAME
    if mc.draggerContext(ctx, q=True, button=True) != 1:
        return
    if not get_plane():
        mc.warning("The sketch plane is gone; fit a new plane first.")
        return

    x, y = _event_position("anchorPoint")
    modifier = mc.draggerContext(ctx, q=True, modifier=True)
    picked = _pick_point(x, y)

    if picked and modifier == "ctrl":
        delete_point(picked)
        mc.select(cl=True)
        return

    if picked:
        mc.select(picked, r=True)
        start = om.MVector(*mc.xform(picked, q=True, t=True, ws=True))
        normal = surface_frame()[3]
        source, direction = _mouse_ray(x, y)
        hit = _intersect(source, direction, start, normal)
        _drag.update(
            point=picked,
            start=start,
            normal=normal,
            offset=(start - hit) if hit is not None else om.MVector(),
            along_normal=modifier == "shift",
        )
        return

    source, direction = _mouse_ray(x, y)
    hit = intersect_plane(source, direction)
    if hit is None:
        mc.warning("The view is edge-on to the sketch plane; rotate the camera.")
        return
    mc.select(add_point(hit), r=True)


def _on_drag() -> None:
    point = _drag.get("point")
    if not point or not mc.objExists(point):
        return
    x, y = _event_position("dragPoint")
    source, direction = _mouse_ray(x, y)
    start, normal = _drag["start"], _drag["normal"]

    if _drag["along_normal"]:
        target = _closest_on_line(source, direction, start, normal)
    else:
        hit = _intersect(source, direction, start, normal)
        target = hit + _drag["offset"] if hit is not None else None
    if target is None:
        return
    mc.xform(point, t=mvec_to_tuple(target), ws=True)
    mc.refresh(currentView=True)


def _on_release() -> None:
    _drag.clear()


def _event_position(flag: str) -> tuple[int, int]:
    """Mouse position in viewport port coordinates (draggerContext space="screen")."""
    x, y, _ = mc.draggerContext(CONTEXT_NAME, q=True, **{flag: True})
    return int(x), int(y)


def _mouse_ray(x: int, y: int) -> tuple[om.MVector, om.MVector]:
    source, direction = om.MPoint(), om.MVector()
    omui.M3dView.active3dView().viewToWorld(x, y, source, direction)
    return om.MVector(source), direction.normal()


def _intersect(source, direction, origin, normal) -> om.MVector | None:
    """Ray / plane hit, or None when the ray runs parallel to the plane."""
    denom = direction * normal
    if abs(denom) < 1e-8:
        return None  # looking at the plane edge-on
    return source + direction * (((origin - source) * normal) / denom)


def _closest_on_line(source, direction, origin, axis) -> om.MVector | None:
    """Point on the line (origin, axis) closest to the mouse ray, or None if they are parallel."""
    w0 = origin - source
    b = axis * direction
    denom = 1.0 - b * b  # both unit vectors
    if denom < 1e-8:
        return None
    t = (b * (direction * w0) - (axis * w0)) / denom
    return origin + axis * t


def _pick_point(x: int, y: int) -> str | None:
    """Closest sketch point within PICK_RADIUS_PX of the cursor, or None."""
    view = omui.M3dView.active3dView()
    best, best_dist = None, PICK_RADIUS_PX**2
    for point in list_points():
        world = om.MPoint(*mc.xform(point, q=True, t=True, ws=True))
        px, py, visible = view.worldToView(world)
        if not visible:
            continue
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist <= best_dist:
            best, best_dist = point, dist
    return best


# ---------------------------------------------------------------------------
# Joint chain through the sketch points

_ORIENT_ATTRS = ("jointOrientX", "jointOrientY", "jointOrientZ")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def create_joints(
    aim_axis: str = DEFAULT_AIM_AXIS,
    up_axis: str = DEFAULT_UP_AXIS,
    up_vector: str = DEFAULT_UP_VECTOR,
    side: str = "",
    name: str = DEFAULT_JOINT_NAME,
) -> list[str]:
    """Build a joint chain through the sketch points, named [side_]{name}_{n}.

    side -- "", "L" or "R"
    """
    aim_index, aim_sign = _parse_axis(aim_axis)
    up_index, up_sign = _parse_axis(up_axis)
    if aim_index == up_index:
        raise RuntimeError("Aim axis and up axis must be different.")

    positions = point_positions()
    if not positions:
        raise RuntimeError("No sketch points. Draw points on the plane first.")

    _, frame_x, plane_y, frame_z = plane_frame()
    _, surface_width, _, surface_normal = surface_frame()
    # "Plane Normal" is whichever axis arrow sticks out of the visible plane
    # (blue Z, or red X after Swap Normal), using that arrow's direction.
    normal_arrow = frame_x if abs(frame_x * surface_normal) > abs(frame_z * surface_normal) else frame_z
    world_up = {"Plane X": frame_x, "Plane Z": frame_z}.get(up_vector, normal_arrow)
    # Used when a bone points straight along the up vector.
    fallback_up = surface_width if abs(world_up * surface_normal) > 0.5 else surface_normal
    rotations = _chain_rotations(positions, world_up, fallback_up, plane_y, aim_index, aim_sign, up_index, up_sign)
    radius = plane_size() * 0.02

    mc.select(cl=True)
    joints = []
    parent_matrix = om.MMatrix()  # identity: root lives in world space
    for index, (position, rotation) in enumerate(zip(positions, rotations)):
        parent_kwargs = {"p": joints[-1]} if joints else {}
        joint = mc.createNode("joint", n=joint_name(side, name or DEFAULT_JOINT_NAME, index + 1), **parent_kwargs)
        joint = mc.ls(joint, long=True)[0]

        world = om.MMatrix(rotation)
        world.setElement(3, 0, position.x)
        world.setElement(3, 1, position.y)
        world.setElement(3, 2, position.z)
        local = om.MTransformationMatrix(world * parent_matrix.inverse())
        orient = local.rotation(asQuaternion=False)  # jointOrient is always XYZ order
        translate = local.translation(om.MSpace.kTransform)

        mc.setAttr(joint + ".translate", translate.x, translate.y, translate.z)
        mc.setAttr(joint + ".jointOrient", *[om.MAngle(a).asDegrees() for a in (orient.x, orient.y, orient.z)])
        mc.setAttr(joint + ".radius", radius)
        mc.addAttr(joint, ln=JOINT_ATTR, at="bool", dv=True)
        expose_joint_orient(joint)
        joints.append(joint)
        parent_matrix = world

    add_to_isolation(joints)
    mc.select(joints[0], r=True)
    return joints


def expose_joint_orient(joint: str) -> None:
    """Make jointOrientX/Y/Z visible and keyable in the Channel Box."""
    for attr in _ORIENT_ATTRS:
        node_attr = "{}.{}".format(joint, attr)
        mc.setAttr(node_attr, cb=True)
        mc.setAttr(node_attr, k=True)


def _parse_axis(axis: str) -> tuple[int, float]:
    return _AXIS_INDEX[axis[-1].upper()], (-1.0 if axis.startswith("-") else 1.0)


def _chain_rotations(positions, world_up, fallback_up, plane_y, aim_index, aim_sign, up_index, up_sign):
    """Rotation (flat 4x4 list) per joint; the last joint copies its parent's."""
    aims = []
    for current, following in zip(positions, positions[1:]):
        aim = following - current
        aims.append(aim.normal() if aim.length() > 1e-6 else None)
    if not aims:
        aims = [plane_y]  # single joint: aim along the plane's long axis
    else:
        aims.append(aims[-1])

    # Coincident points: borrow the nearest valid aim.
    last_valid = next((a for a in aims if a is not None), plane_y)
    for i, aim in enumerate(aims):
        if aim is None:
            aims[i] = last_valid
        else:
            last_valid = aim

    rotations = []
    for aim in aims:
        up = world_up - aim * (world_up * aim)
        if up.length() < 1e-6:
            # Aim is parallel to the up vector: fall back to another plane axis.
            up = fallback_up - aim * (fallback_up * aim)
        up = up.normal()
        rotations.append(_basis(aim * aim_sign, aim_index, up * up_sign, up_index))
    return rotations


def _basis(aim, aim_index, up, up_index) -> list[float]:
    rows = [None, None, None]
    rows[aim_index] = aim
    rows[up_index] = up
    third = 3 - aim_index - up_index
    # Right-handed: X = Y ^ Z, Y = Z ^ X, Z = X ^ Y
    a, b = rows[(third + 1) % 3], rows[(third + 2) % 3]
    rows[third] = (a ^ b).normal()
    flat = []
    for row in rows:
        flat += [row.x, row.y, row.z, 0.0]
    return flat + [0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# UI


def _get_option(key: str, default):
    name = OPTION_PREFIX + key
    return mc.optionVar(q=name) if mc.optionVar(exists=name) else default


def _set_option(key: str, value) -> None:
    name = OPTION_PREFIX + key
    if isinstance(value, bool):
        mc.optionVar(iv=(name, int(value)))
    else:
        mc.optionVar(sv=(name, value))


def _run(func, *args):
    """Run func in one undo chunk, reporting tool errors as warnings."""
    mc.undoInfo(openChunk=True, chunkName="boneSketcher")
    try:
        return func(*args)
    except RuntimeError as exc:
        mc.warning(str(exc))
    finally:
        mc.undoInfo(closeChunk=True)


def _fit_plane() -> str:
    selection = mc.ls(sl=True)
    plane = fit_plane_to_selection()
    store_selection(selection)
    update_cameras(plane)
    mc.select(plane, r=True)
    return plane


def _swap_normal() -> None:
    swap_normal()
    update_cameras()


def ui():
    win = "bone_sketcher_win"

    def _option_menu(label, items, current):
        menu = mc.optionMenuGrp(l=label, cal=[1, "left"], cw2=[80, 200])
        for item in items:
            mc.menuItem(l=item)
        if current in items:
            mc.optionMenuGrp(menu, e=True, v=current)
        return menu

    def _refresh_draw_button(*_):
        if mc.button(draw_btn, exists=True):
            mc.button(draw_btn, e=True, l="Stop Drawing" if is_drawing() else "Draw Points")

    def _draw_triggered(*_):
        if is_drawing():
            leave_draw_tool()
        else:
            try:
                enter_draw_tool()
            except RuntimeError as exc:
                mc.warning(str(exc))
        _refresh_draw_button()

    def _delete_sketch_triggered(*_):
        leave_draw_tool()
        restore_viewports()
        _run(delete_sketch)

    def _create_joints_triggered(*_):
        side_label = mc.optionMenuGrp(side_omg, q=True, v=True)
        options = {
            "aim_axis": mc.optionMenuGrp(aim_omg, q=True, v=True),
            "up_axis": mc.optionMenuGrp(up_omg, q=True, v=True),
            "up_vector": mc.optionMenuGrp(up_vec_omg, q=True, v=True),
            "side": SIDE_CHOICES[side_label],
            "name": mc.textFieldGrp(name_tfg, q=True, tx=True).strip("_ ") or DEFAULT_JOINT_NAME,
        }
        delete_after = mc.checkBox(delete_cb, q=True, v=True)
        _set_option("aimAxis", options["aim_axis"])
        _set_option("upAxis", options["up_axis"])
        _set_option("upVector", options["up_vector"])
        _set_option("jointSide", side_label)
        _set_option("jointName", options["name"])
        _set_option("deleteSketch", delete_after)

        def _create():
            created = create_joints(**options)
            if delete_after:
                restore_viewports()
                delete_sketch()
                mc.select(created[0], r=True)

        leave_draw_tool()
        _run(_create)

    if mc.window(win, exists=True):
        mc.deleteUI(win)

    mc.window(win, t="Bone Sketcher")
    mc.columnLayout(adj=True, rs=2, cat=["both", 1])

    # 1. Sketch plane
    mc.frameLayout(l="1. Sketch Plane", mw=4, mh=4)
    mc.columnLayout(adj=True, rs=2)
    mc.button(
        l="Fit Plane to Selection",
        c=lambda *_: _run(_fit_plane),
        h=30,
        ann="Select an object or components, then click. Y follows the longest side.\n"
        "With a plane already present, this refits it and keeps the points.",
    )
    mc.rowColumnLayout(nc=2, cw=[(1, 140), (2, 140)], cs=[(2, 2)])
    mc.button(
        l="Rotate 90",
        c=lambda *_: _run(rotate_plane_90),
        ann="Roll the root (plane and handles) 90 degrees about its Y axis (for round shapes\n"
        "where the fit is ambiguous). Placed points stay where they are.",
    )
    mc.button(
        l="Swap Normal Z <> X",
        c=lambda *_: _run(_swap_normal),
        ann="Swap the plane's X and Z axes. The plane itself stays where it is;\n"
        "only the axis arrows turn, so the red X becomes the one sticking out of the plane.\n"
        "Press again to swap back.",
    )
    mc.setParent("..")
    mc.button(
        l="Flip Root <> Tip",
        c=lambda *_: _run(flip_direction),
        ann="Swap the root and the tip (aim handle) so the plane's Y points the other way.\n"
        "The plane covers the same area and the normal is kept. Placed points stay where they are.",
    )
    mc.button(
        l="Isolate Selected + Sketch",
        c=lambda *_: _run(toggle_isolate),
        ann="Toggle: in the active viewport show only the components the plane was fitted to,\n"
        "the sketch plane / points and the joints made by Bone Sketcher.",
    )
    mc.button(l="Delete Sketch", c=_delete_sketch_triggered, ann="Delete the plane and all sketch points.")
    mc.setParent("..")
    mc.setParent("..")

    # Views
    mc.frameLayout(l="Views", mw=4, mh=4)
    mc.rowColumnLayout(nc=3, cw=[(1, 93), (2, 93), (3, 93)], cs=[(2, 2), (3, 2)], rs=[(1, 2)])
    for view in view_names() + ["persp"]:
        mc.button(
            l=view.capitalize(),
            c=partial(lambda v, *_: _run(look_through, v), view),
            ann=(
                "Back to the viewport's previous camera."
                if view == "persp"
                else "Look through the plane's {} camera in the active viewport.".format(view)
            ),
        )
    mc.setParent("..")
    mc.setParent("..")

    # 2. Points
    mc.frameLayout(l="2. Points", mw=4, mh=4)
    mc.columnLayout(adj=True, rs=2)
    draw_btn = mc.button(l="Draw Points", c=_draw_triggered, h=30)
    mc.text(
        l="Click the plane to add a point.\n"
        "Drag a point to slide it along the plane.\n"
        "Shift+drag a point to lift it off the plane.\n"
        "Ctrl+click a point to delete it.",
        al="left",
        en=False,
    )
    mc.button(l="Delete Last Point", c=lambda *_: _run(delete_last_point))
    mc.setParent("..")
    mc.setParent("..")

    # 3. Joints
    naming_tip = "Joints are named [L_|R_]<Name>_<n>, e.g. L_arm_1."
    mc.frameLayout(l="3. Joints", mw=4, mh=4)
    mc.columnLayout(adj=True, rs=2)
    side_omg = _option_menu("Side:", list(SIDE_CHOICES), _get_option("jointSide", DEFAULT_JOINT_SIDE))
    mc.optionMenuGrp(side_omg, e=True, ann=naming_tip)
    name_tfg = mc.textFieldGrp(
        l="Name:",
        tx=_get_option("jointName", DEFAULT_JOINT_NAME),
        cal=[1, "left"],
        cw2=[80, 200],
        ann=naming_tip,
    )
    aim_omg = _option_menu("Aim axis:", AXIS_CHOICES, _get_option("aimAxis", DEFAULT_AIM_AXIS))
    up_omg = _option_menu("Up axis:", AXIS_CHOICES, _get_option("upAxis", DEFAULT_UP_AXIS))
    up_vec_omg = _option_menu("Up vector:", UP_VECTOR_CHOICES, _get_option("upVector", DEFAULT_UP_VECTOR))
    mc.optionMenuGrp(
        up_vec_omg,
        e=True,
        ann="Direction every joint's up axis points along.\n"
        "Plane Normal: perpendicular to the visible plane (even after Swap Normal).\n"
        "Plane X / Plane Z: the red / blue axis arrows as currently shown.",
    )
    delete_cb = mc.checkBox(l="Delete sketch after creating joints", v=bool(_get_option("deleteSketch", True)))
    mc.button(l="Create Joints", c=_create_joints_triggered, h=40)
    mc.setParent("..")
    mc.setParent("..")

    # Keeps the Draw button label in sync when the user switches tools in Maya.
    mc.scriptJob(event=["ToolChanged", _refresh_draw_button], p=win)
    _refresh_draw_button()

    mc.showWindow(win)
    mc.window(win, e=True, w=300, h=100, rtf=True)
