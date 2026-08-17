from __future__ import annotations
import numpy as np
import trimesh
from trimesh.viewer import SceneViewer
from PIL import Image
import ctypes
import pyglet
from pyglet.gl import GL_REPEAT, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T

def maximize_viewer(dt):
    SW_MAXIMIZE = 3

    hwnd = ctypes.windll.user32.GetForegroundWindow()

    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)

def make_box(
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> trimesh.Trimesh:
    """Create a box at the requested center."""
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh
   
def apply_box_texture(mesh, texture_path, scale=1.0):
    texture = Image.open(texture_path)

    material = trimesh.visual.texture.SimpleMaterial(
        image=texture,
        kwargs={
            "tex_parameter": {
                GL_TEXTURE_WRAP_S: GL_REPEAT,
                GL_TEXTURE_WRAP_T: GL_REPEAT,
            }
        },
    )

    # Work on a copy so the original mesh is preserved
    mesh = mesh.copy()

    # Each face gets independent vertices.
    # This is important for correct UVs at sharp corners.
    mesh.unmerge_vertices()

    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals

    # Vertex positions for every triangle:
    # shape = (n_faces, 3, 3)
    face_vertices = vertices[faces]

    # Dominant normal axis for each face:
    # 0 = X, 1 = Y, 2 = Z
    axes = np.argmax(np.abs(normals), axis=1)

    uvs = np.zeros((len(vertices), 2), dtype=np.float32)

    # X-facing surfaces: use Y/Z
    mask = axes == 0
    if np.any(mask):
        fv = face_vertices[mask]
        uv = np.stack(
            (
                fv[:, :, 1] * scale,
                fv[:, :, 2] * scale,
            ),
            axis=-1,
        )
        uvs[faces[mask]] = uv

    # Y-facing surfaces: use X/Z
    mask = axes == 1
    if np.any(mask):
        fv = face_vertices[mask]
        uv = np.stack(
            (
                fv[:, :, 0] * scale,
                fv[:, :, 2] * scale,
            ),
            axis=-1,
        )
        uvs[faces[mask]] = uv

    # Z-facing surfaces: use X/Y
    mask = axes == 2
    if np.any(mask):
        fv = face_vertices[mask]
        uv = np.stack(
            (
                fv[:, :, 0] * scale,
                fv[:, :, 1] * scale,
            ),
            axis=-1,
        )
        uvs[faces[mask]] = uv

    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uvs,
        material=material,
    )

    return mesh

def create_four_walled_house(width=5.0, depth=7.0, height=3.5, wall_thickness=0.2, roof_height=2.0):
    """
    Assembles a house using 4 individual thin wall primitives and a roof.
    """
    half_w = width / 2.0
    half_d = depth / 2.0
    z_pos = height / 2.0
    
    brick_path = "assets//textures//building_materials//bricks//Bricks097_1K-JPG//Bricks097_1K-JPG_Color.jpg"
    door_path  = "assets//textures//building_materials//door//wood_panel_door_glass.png"
    roof_path  = "assets//textures//building_materials//roof//roof_shingles.png"
    
    # ----------------------------------------------------
    # 1. CREATE INDIVIDUAL WALLS
    # ----------------------------------------------------
    # Front Wall (along the bottom Y edge)
    front_wall = trimesh.creation.box(extents=[width, wall_thickness, height])
    front_wall.apply_translation([0, -half_d + (wall_thickness / 2.0), z_pos])
    
    # Back Wall (along the top Y edge)
    back_wall = trimesh.creation.box(extents=[width, wall_thickness, height])
    back_wall.apply_translation([0, half_d - (wall_thickness / 2.0), z_pos])
    
    # Left Wall (along the left X edge)
    # Note: Depth is reduced by 2*thickness to cleanly fit between front and back walls
    side_depth = depth - (2 * wall_thickness)
    left_wall = trimesh.creation.box(extents=[wall_thickness, side_depth, height])
    left_wall.apply_translation([-half_w + (wall_thickness / 2.0), 0, z_pos])
    
    # Right Wall (along the right X edge)
    right_wall = trimesh.creation.box(extents=[wall_thickness, side_depth, height])
    right_wall.apply_translation([half_w - (wall_thickness / 2.0), 0, z_pos])

    # ----------------------------------------------------
    # 2. APPLY INDIVIDUAL TEXTURES
    # ----------------------------------------------------
    # You can pass different image paths to each wall if desired
    front_wall = apply_box_texture(front_wall, brick_path, scale=1.5)
    back_wall = apply_box_texture(back_wall, brick_path, scale=1.5)
    left_wall = apply_box_texture(left_wall, brick_path, scale=1.5)
    right_wall = apply_box_texture(right_wall, brick_path, scale=1.5)

    # ----------------------------------------------------
    # 3. ADD FIXTURES (Door & Windows)
    # ----------------------------------------------------
    door_w, door_h = 1.0, 2.1
    door_mesh = trimesh.creation.box(extents=[door_w, 0.05, door_h])
    # Sit the door slightly proud of the front wall skin to avoid Z-fighting glitches
    door_mesh.apply_translation([0, -half_d - 0*wall_thickness + 0.01, door_h / 2.0])
    # door_mesh = apply_box_texture(door_mesh, door_path, scale=0.)
    door_mesh.visual.face_colors = [120, 80, 54, 255] # Brown Door

    win_w, win_h = 1.2, 1.2
    glass_left = trimesh.creation.box(extents=[0.05, win_w, win_h])
    glass_left.apply_translation([-half_w + 0*wall_thickness + 0.01, 0, height * 0.6])
    glass_left.visual.face_colors = [100, 149, 237, 150]

    # ----------------------------------------------------
    # 4. ROOF GENERATION
    # ----------------------------------------------------
    roof_vertices = np.array([
        [-half_w, -half_d, 0], [half_w, -half_d, 0], 
        [half_w, half_d, 0], [-half_w, half_d, 0],
        [0, -half_d, roof_height], [0, half_d, roof_height]
    ])
    roof_faces = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 3, 5], [3, 0, 5], [0, 4, 5], [0, 3, 2], [0, 2, 1]])
    roof_mesh = trimesh.Trimesh(vertices=roof_vertices, faces=roof_faces)
    roof_mesh.apply_translation([0, 0, height])
    roof_mesh = apply_box_texture(roof_mesh, roof_path, scale=1.5)
    # roof_mesh.visual.face_colors = [150, 50, 50, 255]

    # Compile all independent panels into the final multi-object scene graph
    return trimesh.Scene([front_wall, back_wall, left_wall, right_wall, door_mesh, glass_left, roof_mesh])

def create_gable_roof(
    house_width: float,
    house_depth: float,
    wall_height: float,
    roof_height: float,
    roof_thickness: float = 0.15,
) -> trimesh.Trimesh:
    """
    Create a simple gable roof.

    Ridge runs along Y.
    """

    # Roof cross-section vertices
    vertices = np.array([
        [0, 0, wall_height],
        [house_width, 0, wall_height],
        [house_width / 2, 0, wall_height + roof_height],

        [0, house_depth, wall_height],
        [house_width, house_depth, wall_height],
        [house_width / 2, house_depth, wall_height + roof_height],
    ], dtype=float)

    # Two roof planes + underside surfaces
    faces = np.array([
        # Front slope
        [0, 1, 2],

        # Back slope
        [3, 5, 4],

        # Left underside
        [0, 3, 4],
        [0, 4, 1],

        # Right underside
        [1, 4, 5],
        [1, 5, 2],

        # Left roof end
        [0, 2, 5],
        [0, 5, 3],

        # Right roof end
        [0, 3, 5],
        [0, 5, 2],
    ], dtype=int)

    roof = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
    )

    return roof

def create_window(
    width: float,
    height: float,
    thickness: float = 0.08,
) -> trimesh.Trimesh:

    # Glass
    glass = make_box(
                (
                    width,
                    thickness,
                    height,
                ),
                (
                    0,
                    0,
                    height / 2,
                ),
            )
    glass.visual.face_colors = [100, 149, 237, 150]
    return glass

def create_door(
    width: float,
    height: float,
    thickness: float = 0.08,
) -> trimesh.Trimesh:

    frame = 0.10

    parts = []

    # Door slab
    parts.append(
        make_box(
            (
                width,
                thickness,
                height,
            ),
            (
                0,
                0,
                height / 2,
            ),
        )
    )

    # Left frame
    parts.append(
        make_box(
            (
                frame,
                thickness * 1.5,
                height,
            ),
            (
                -width / 2,
                0,
                height / 2,
            ),
        )
    )

    # Right frame
    parts.append(
        make_box(
            (
                frame,
                thickness * 1.5,
                height,
            ),
            (
                width / 2,
                0,
                height / 2,
            ),
        )
    )

    # Top frame
    parts.append(
        make_box(
            (
                width,
                thickness * 1.5,
                frame,
            ),
            (
                0,
                0,
                height,
            ),
        )
    )

    return trimesh.util.concatenate(parts)



# ================================================================
# WALL WITH OPENINGS
# ================================================================

def create_wall_with_openings(
    length: float,
    height: float,
    thickness: float,
    openings: list[dict],
    axis: str,
    position: tuple[float, float],
    z_min: float = 0.0,
) -> trimesh.Trimesh:
    """
    Create a wall with rectangular openings.

    For axis == "x":
        Wall runs along X.
        position = (y, 0)

    For axis == "y":
        Wall runs along Y.
        position = (x, 0)

    Opening format:

        {
            "type": "window" | "door",
            "offset": 3.0,
            "z": 0.9,
            "width": 1.5,
            "height": 1.4,
        }

    offset is distance along the wall from its minimum coordinate.
    """

    wall_parts = []

    # Sort openings along the wall
    openings = sorted(openings, key=lambda o: o["offset"])

    current = 0.0

    for opening in openings:

        offset = opening["offset"]
        width = opening["width"]
        wz0 = opening["z"]
        wz1 = wz0 + opening["height"]

        opening_start = offset - width / 2
        opening_end = offset + width / 2

        # ----------------------------------------------------------
        # WALL TO LEFT / FRONT OF OPENING
        # ----------------------------------------------------------

        if opening_start > current:

            segment_length = opening_start - current

            if axis == "x":
                center = (
                    current + segment_length / 2,
                    position[0],
                    z_min + height / 2,
                )

                size = (
                    segment_length,
                    thickness,
                    height,
                )

            else:
                center = (
                    position[0],
                    current + segment_length / 2,
                    z_min + height / 2,
                )

                size = (
                    thickness,
                    segment_length,
                    height,
                )

            wall_parts.append(
                make_box(size, center)
            )

        # ----------------------------------------------------------
        # WALL BELOW OPENING
        # ----------------------------------------------------------

        if wz0 > z_min:

            segment_height = wz0 - z_min

            if axis == "x":
                center = (
                    offset,
                    position[0],
                    z_min + segment_height / 2,
                )

                size = (
                    width,
                    thickness,
                    segment_height,
                )

            else:
                center = (
                    position[0],
                    offset,
                    z_min + segment_height / 2,
                )

                size = (
                    thickness,
                    width,
                    segment_height,
                )

            wall_parts.append(
                make_box(size, center)
            )

        # ----------------------------------------------------------
        # WALL ABOVE OPENING
        # ----------------------------------------------------------

        if wz1 < z_min + height:

            segment_height = z_min + height - wz1

            if axis == "x":
                center = (
                    offset,
                    position[0],
                    wz1 + segment_height / 2,
                )

                size = (
                    width,
                    thickness,
                    segment_height,
                )

            else:
                center = (
                    position[0],
                    offset,
                    wz1 + segment_height / 2,
                )

                size = (
                    thickness,
                    width,
                    segment_height,
                )

            wall_parts.append(
                make_box(size, center)
            )

        current = opening_end

    # --------------------------------------------------------------
    # FINAL WALL SECTION
    # --------------------------------------------------------------

    if current < length:

        segment_length = length - current

        if axis == "x":

            center = (
                current + segment_length / 2,
                position[0],
                z_min + height / 2,
            )

            size = (
                segment_length,
                thickness,
                height,
            )

        else:

            center = (
                position[0],
                current + segment_length / 2,
                z_min + height / 2,
            )

            size = (
                thickness,
                segment_length,
                height,
            )

        wall_parts.append(
            make_box(size, center)
        )

    return trimesh.util.concatenate(wall_parts)


# ================================================================
# PLACE WINDOW / DOOR ON WALL
# ================================================================

def add_opening_geometry(
    scene: trimesh.Scene,
    opening: dict,
    wall: str,
    house_width: float,
    house_depth: float,
    wall_thickness: float,
):
    """
    Add visual window/door geometry to the opening.

    wall:
        "front"
        "back"
        "left"
        "right"
    """

    width = opening["width"]
    height = opening["height"]
    offset = opening["offset"]
    thickness = opening["thickness"]
    z = opening["z"]

    if opening["type"] == "window":

        opening_mesh = create_window(
            width,
            height,
            thickness
        )

    elif opening["type"] == "door":

        opening_mesh = create_door(
            width,
            height,
            thickness
        )

    else:

        raise ValueError(
            f"Unknown opening type: {opening['type']}"
        )

    # --------------------------------------------------------------
    # FRONT
    # --------------------------------------------------------------

    if wall == "front":

        # opening_mesh.apply_transform(
        #     trimesh.transformations.rotation_matrix(
        #         np.pi / 2,
        #         [1, 0, 0],
        #     )
        # )

        opening_mesh.apply_translation(
            (
                offset,
                thickness/2,
                z,
            )
        )

    # --------------------------------------------------------------
    # BACK
    # --------------------------------------------------------------

    elif wall == "back":

        # opening_mesh.apply_transform(
        #     trimesh.transformations.rotation_matrix(
        #         -np.pi / 2,
        #         [1, 0, 0],
        #     )
        # )

        opening_mesh.apply_translation(
            (
                offset,
                house_depth + wall_thickness / 2 + 0.02,
                z,
            )
        )

    # --------------------------------------------------------------
    # LEFT
    # --------------------------------------------------------------

    elif wall == "left":

        opening_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.pi / 2,
                [0, 0, 1],
            )
        )

        opening_mesh.apply_translation(
            (
                -wall_thickness / 2 - 0.02,
                offset,
                z,
            )
        )

    # --------------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------------

    elif wall == "right":

        opening_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                -np.pi / 2,
                [0, 0, 1],
            )
        )

        opening_mesh.apply_translation(
            (
                house_width + wall_thickness / 2 + 0.02,
                offset,
                z,
            )
        )

    scene.add_geometry(
        opening_mesh,
        node_name=f"{wall}_{opening['type']}",
    )


# --------------------------------------------------------
# EXECUTION ROUTINE
# --------------------------------------------------------
def create_house(
    width: float = 10.0,
    depth: float = 8.0,
    wall_height: float = 3.0,
    wall_thickness: float = 0.20,
    roof_height: float = 2.0,
    wall_path: str = "assets//textures//building_materials//bricks//Bricks097_1K-JPG//Bricks097_1K-JPG_Color.jpg",
    door_path: str = "assets//textures//building_materials//door//wood_panel_door_glass.png",
    roof_path: str = "assets//textures//building_materials//roof//roof_shingles.png"
) -> trimesh.Scene:
    scene = trimesh.Scene()

    # ============================================================
    # OPENINGS
    # ============================================================

    front_openings = [
        {
            "type": "door",
            "offset": width / 2,
            "z": 0.0,
            "width": 1.0,
            "height": 2.2,
            "thickness": 0.1
        },
        {
            "type": "window",
            "offset": 2.0,
            "z": 1.0,
            "width": 1.5,
            "height": 1.3,
            "thickness": 0.08
        },
        {
            "type": "window",
            "offset": width - 2.0,
            "z": 1.0,
            "width": 1.5,
            "height": 1.3,
            "thickness": 0.08
        },
    ]

    back_openings = [
        {
            "type": "window",
            "offset": width / 2,
            "z": 1.0,
            "width": 2.0,
            "height": 1.3,
            "thickness": 0.08
        }
    ]

    left_openings = [
        {
            "type": "window",
            "offset": depth / 2,
            "z": 1.0,
            "width": 1.5,
            "height": 1.3,
            "thickness": 0.08
        }
    ]

    right_openings = [
        {
            "type": "window",
            "offset": depth / 2,
            "z": 1.0,
            "width": 1.5,
            "height": 1.3,
            "thickness": 0.08
        }
    ]

    # ============================================================
    # FRONT WALL
    # ============================================================

    front = create_wall_with_openings(
        length=width,
        height=wall_height,
        thickness=wall_thickness,
        openings=front_openings,
        axis="x",
        position=(0, 0),
    )

    front.apply_translation(
        (
            0,
            0+wall_thickness/2,
            0,
        )
    )
    front = apply_box_texture(front, wall_path, scale=1.5)
    scene.add_geometry(
        front,
        node_name="front_wall",
    )

    # ============================================================
    # BACK WALL
    # ============================================================

    back = create_wall_with_openings(
        length=width,
        height=wall_height,
        thickness=wall_thickness,
        openings=back_openings,
        axis="x",
        position=(0, 0),
    )

    back.apply_translation(
        (
            0,
            depth-wall_thickness/2,
            0,
        )
    )
    back = apply_box_texture(back, wall_path, scale=1.5)
    scene.add_geometry(
        back,
        node_name="back_wall",
    )

    # ============================================================
    # LEFT WALL
    # ============================================================

    left = create_wall_with_openings(
        length=depth-2*wall_thickness,
        height=wall_height,
        thickness=wall_thickness,
        openings=left_openings,
        axis="y",
        position=(0, 0),
    )

    left.apply_translation(
        (
            wall_thickness/2,
            wall_thickness,
            0,
        )
    )
    left = apply_box_texture(left, wall_path, scale=1.5)
    scene.add_geometry(
        left,
        node_name="left_wall",
    )

    # ============================================================
    # RIGHT WALL
    # ============================================================

    right = create_wall_with_openings(
        length=depth-2*wall_thickness,
        height=wall_height,
        thickness=wall_thickness,
        openings=right_openings,
        axis="y",
        position=(0, 0),
    )

    right.apply_translation(
        (
            width-wall_thickness/2,
            wall_thickness,
            0,
        )
    )
    right = apply_box_texture(right, wall_path, scale=1.5)
    scene.add_geometry(
        right,
        node_name="right_wall",
    )

    # ============================================================
    # FLOOR
    # ============================================================

    floor = make_box(
        (
            width,
            depth,
            0.2,
        ),
        (
            width / 2,
            depth / 2,
            -0.1,
        ),
    )

    scene.add_geometry(
        floor,
        node_name="floor",
    )

    # ============================================================
    # ROOF
    # ============================================================

    roof = create_gable_roof(
        house_width=width,
        house_depth=depth,
        wall_height=wall_height,
        roof_height=roof_height,
    )
    roof = apply_box_texture(roof, roof_path, scale=1.5)
    scene.add_geometry(
        roof,
        node_name="roof",
    )

    # ============================================================
    # WINDOWS / DOORS
    # ============================================================

    for i, opening in enumerate(front_openings):
        add_opening_geometry(
            scene,
            opening,
            "front",
            width,
            depth,
            wall_thickness,
        )

    for i, opening in enumerate(back_openings):
        add_opening_geometry(
            scene,
            opening,
            "back",
            width,
            depth,
            wall_thickness,
        )

    for i, opening in enumerate(left_openings):
        add_opening_geometry(
            scene,
            opening,
            "left",
            width,
            depth,
            wall_thickness,
        )

    for i, opening in enumerate(right_openings):
        add_opening_geometry(
            scene,
            opening,
            "right",
            width,
            depth,
            wall_thickness,
        )

    return scene    
    
if __name__ == "__main__":
    house_scene = create_house(
        width=10.0,
        depth=8.0,
        wall_height=3.0,
        wall_thickness=0.20,
        roof_height=2.0,
    )
    # house_scene = create_four_walled_house(width=5.0, depth=7.0, height=3.5, roof_height=2.0)
    
    # Launch standard interactive 3D pyglet viewport frame
    viewer = SceneViewer(
        house_scene,
        start_loop=False
    )
    
    pyglet.clock.schedule_once(maximize_viewer, 0.1)
    pyglet.app.run()