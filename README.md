# Bone Sketcher

A Maya tool for sketching joint chains directly on your model.

Fit a plane to a mesh or a component selection, click points on it, and build a
joint chain through them. Every joint aims at the next one and shares the same up
vector, so the whole chain lies on a single rotation plane, ready for IK.

## Requirements

- Maya 2022 or newer (Python 3)
- No other dependencies; it is a single file.

## Installation

Put `bone_sketcher.py` in a folder on Maya's Python path (for example your
`maya/scripts` folder), or add its folder to `sys.path`. Then run this in a
Python tab of the Script Editor:

```python
import sys
import importlib

path = r"C:\path\to\bone_sketcher"  # folder that contains bone_sketcher.py
if path not in sys.path:
    sys.path.insert(0, path)

import bone_sketcher
importlib.reload(bone_sketcher)
bone_sketcher.ui()
```

Drag the snippet to a shelf to keep it as a button.

## Workflow

### 1. Sketch Plane

Select a mesh or any mesh components (vertices, edges, faces), then click
**Fit Plane to Selection**. The plane's Y axis follows the longest side of the
selection, and its normal faces the active camera.

| Control | What it does |
| --- | --- |
| Root (white ring) | Moves and rotates the whole plane rig. |
| Aim handle (green) | The plane's Y axis always aims at it. |
| Up handle (blue) | Rolls the plane; its Z axis points toward it. |
| `planeWidth` / `gizmoScale` attributes on the root | Widen the plane and resize the handles. |

- **Rotate 90**: rolls the plane 90° about its Y axis. Use it on round shapes where
  the fitted orientation is ambiguous.
- **Swap Normal Z <> X**: swaps the plane's X and Z axes without moving the plane.
- **Flip Root <> Tip**: swaps the root and the aim handle so the plane's Y points
  the other way. The plane covers the same area, the normal and roll are kept,
  and points already drawn stay where they are.
- **Isolate Selected + Sketch**: shows only the fitted components, the sketch and the
  joints made by the tool. Click again to turn it off.
- **Views**: orthographic cameras locked to the plane (Top, Bottom, Front, Back,
  Side). **Persp** goes back to the previous camera.

Refitting an existing plane keeps the points already drawn.

### 2. Points

Click **Draw Points**, then in the viewport:

| Action | Result |
| --- | --- |
| Click the plane | Add a point |
| Drag a point | Slide it along the plane |
| Shift + drag a point | Move it off the plane, along the normal |
| Ctrl + click a point | Delete it |

Joints are shown in X-ray while drawing, so points stay visible through the mesh.

### 3. Joints

| Option | Description |
| --- | --- |
| Side | None, Left or Right |
| Name | Base name of the chain |
| Aim axis | Joint axis that points down the chain (default `+Y`) |
| Up axis | Joint axis that points along the up vector (default `+X`) |
| Up vector | `Plane Normal` (perpendicular to the visible plane), `Plane X` (red arrow) or `Plane Z` (blue arrow) |
| Delete sketch after creating joints | Clean up the plane and points afterwards |

Click **Create Joints**. Joint orients are set and show in the Channel Box.
Rotations are left at zero.

## Naming

Joints are named `[L_|R_]<name>_<n>`:

| Side | Name | Result |
| --- | --- | --- |
| None | `spine` | `spine_1`, `spine_2`, ... |
| Left | `arm` | `L_arm_1`, `L_arm_2`, ... |
| Right | `leg` | `R_leg_1`, `R_leg_2`, ... |

## Undo

Every button runs as a single undo step. The tool keeps no state in Python;
everything it uses lives in the scene, so undo and redo work normally.
