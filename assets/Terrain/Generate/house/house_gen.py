from __future__ import annotations
import numpy as np
import trimesh
from trimesh.viewer import SceneViewer
from PIL import Image
import ctypes
import pyglet
from pyglet.gl import GL_REPEAT, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE

def maximize_viewer(dt):
    SW_MAXIMIZE = 3

    hwnd = ctypes.windll.user32.GetForegroundWindow()

    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)

class HOUSE_GEN():
    def __init__(self,width=10.0, depth=8.0, wall_height=3.0, wall_thickness=0.20,roof_height=2.0):
        self.scene = self.create_house(
            width,
            depth,
            wall_height,
            wall_thickness,
            roof_height,
        )

    def make_box(self,
        size: tuple[float, float, float],
        center: tuple[float, float, float],
    ) -> trimesh.Trimesh:
        """Create a box at the requested center."""
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        return mesh
    
    def apply_box_texture(self, mesh, texture_path, scale=1.0, style="repeat"):
        """
        Apply a box-projected texture to a mesh with optional repeat or clamp mode.

        Parameters:
            mesh (trimesh.Trimesh): The mesh to texture.
            texture_path (str): Path to the texture image.
            scale (float): UV scaling factor (higher = more repeats if style='repeat').
            style (str): 'repeat' or 'clamp' for texture wrapping.
        """
        # Load texture image
        try:
            texture = Image.open(texture_path)
        except Exception as e:
            raise ValueError(f"Failed to load texture '{texture_path}': {e}")

        # Choose wrapping mode
        if style.lower() == "repeat":
            tex_params = {
                GL_TEXTURE_WRAP_S: GL_REPEAT,
                GL_TEXTURE_WRAP_T: GL_REPEAT
            }
        else:
            tex_params = {
                GL_TEXTURE_WRAP_S: GL_CLAMP_TO_EDGE,
                GL_TEXTURE_WRAP_T: GL_CLAMP_TO_EDGE
            }
            
        # Create material
        material = trimesh.visual.texture.SimpleMaterial(
            image=texture,
            tex_params=tex_params
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
        
        def normalize_uv(coords):
            """
            Normalize projected coordinates independently in U and V
            so the texture occupies exactly [0,1] across the surface.
            """
            uv_min = coords.min(axis=0)
            uv_max = coords.max(axis=0)

            extent = uv_max - uv_min
            extent[extent < 1e-8] = 1.0

            return (coords - uv_min) / extent

        # X-facing surfaces: use Y/Z for UV
        mask = axes == 0
        if np.any(mask):
            fv = face_vertices[mask]

            coords = np.stack(
                (
                    fv[:, :, 1],
                    fv[:, :, 2],
                ),
                axis=-1,
            )

            if style.lower() == "clamp":
                # Normalize each surface to [0,1]
                coords_flat = coords.reshape(-1, 2)
                coords_flat = normalize_uv(coords_flat)
                uv = coords_flat.reshape(coords.shape)
            else:
                uv = coords * scale

            uvs[faces[mask]] = uv

        # Y-facing surfaces: use X/Z for UV
        mask = axes == 1
        if np.any(mask):
            fv = face_vertices[mask]

            coords = np.stack(
                (
                    fv[:, :, 0],
                    fv[:, :, 2],
                ),
                axis=-1,
            )

            if style.lower() == "clamp":
                coords_flat = coords.reshape(-1, 2)
                coords_flat = normalize_uv(coords_flat)
                uv = coords_flat.reshape(coords.shape)
            else:
                uv = coords * scale

            uvs[faces[mask]] = uv
            
        # Z-facing surfaces: use X/Y for UV
        mask = axes == 2
        if np.any(mask):
            fv = face_vertices[mask]

            coords = np.stack(
                (
                    fv[:, :, 0],
                    fv[:, :, 1],
                ),
                axis=-1,
            )

            if style.lower() == "clamp":
                coords_flat = coords.reshape(-1, 2)
                coords_flat = normalize_uv(coords_flat)
                uv = coords_flat.reshape(coords.shape)
            else:
                uv = coords * scale

            uvs[faces[mask]] = uv
            
        # Assign UVs and material to mesh
        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=uvs,
            image=texture,
            material=material
        )

        return mesh

    def create_gable_roof(
        self,
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
        self,
        width: float,
        height: float,
        thickness: float = 0.08,
    ) -> trimesh.Trimesh:

        # Glass
        glass = self.make_box(
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
        self,
        width: float,
        height: float,
        thickness: float = 0.08,
        texture: str = ""
    ) -> trimesh.Trimesh:

        frame = 0.10

        parts = []

        # Door slab
        parts.append(
            self.make_box(
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
        if texture != "":
            parts[0] = self.apply_box_texture(parts[0], texture, 1.0, "clamp")

        # # Left frame
        # parts.append(
        #     make_box(
        #         (
        #             frame,
        #             thickness * 1.5,
        #             height,
        #         ),
        #         (
        #             -width / 2,
        #             0,
        #             height / 2,
        #         ),
        #     )
        # )

        # # Right frame
        # parts.append(
        #     make_box(
        #         (
        #             frame,
        #             thickness * 1.5,
        #             height,
        #         ),
        #         (
        #             width / 2,
        #             0,
        #             height / 2,
        #         ),
        #     )
        # )

        # # Top frame
        # parts.append(
        #     make_box(
        #         (
        #             width,
        #             thickness * 1.5,
        #             frame,
        #         ),
        #         (
        #             0,
        #             0,
        #             height,
        #         ),
        #     )
        # )

        return trimesh.util.concatenate(parts)



    # ================================================================
    # WALL WITH OPENINGS
    # ================================================================

    def create_wall_with_openings(
        self,
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
                    self.make_box(size, center)
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
                    self.make_box(size, center)
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
                    self.make_box(size, center)
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
                self.make_box(size, center)
            )

        return trimesh.util.concatenate(wall_parts)


    # ================================================================
    # PLACE WINDOW / DOOR ON WALL
    # ================================================================

    def add_opening_geometry(
        self,
        scene: trimesh.Scene,
        opening: dict,
        wall: str,
        house_width: float,
        house_depth: float,
        wall_thickness: float,
        num: int
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

            opening_mesh = self.create_window(
                width,
                height,
                thickness
            )

        elif opening["type"] == "door":

            opening_mesh = self.create_door(
                width,
                height,
                thickness,
                opening['texture']
            )

        else:

            raise ValueError(
                f"Unknown opening type: {opening['type']}"
            )

        # --------------------------------------------------------------
        # FRONT
        # --------------------------------------------------------------

        if wall == "front":
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
                    house_depth - thickness / 2,
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
                    thickness/2,
                    offset + wall_thickness,
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
                    house_width - thickness/2,
                    offset + wall_thickness,
                    z,
                )
            )

        scene.add_geometry(
            opening_mesh,
            node_name=f"{wall}_{opening['type']}_{num}",
        )


    # --------------------------------------------------------
    # EXECUTION ROUTINE
    # --------------------------------------------------------
    def create_house(
        self,
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
                "thickness": 0.1,
                "texture": door_path
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

        front = self.create_wall_with_openings(
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
        front = self.apply_box_texture(front, wall_path, scale=1.5)
        scene.add_geometry(
            front,
            node_name="front_wall",
        )

        # ============================================================
        # BACK WALL
        # ============================================================

        back = self.create_wall_with_openings(
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
        back = self.apply_box_texture(back, wall_path, scale=1.5)
        scene.add_geometry(
            back,
            node_name="back_wall",
        )

        # ============================================================
        # LEFT WALL
        # ============================================================

        left = self.create_wall_with_openings(
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
        left = self.apply_box_texture(left, wall_path, scale=1.5)
        scene.add_geometry(
            left,
            node_name="left_wall",
        )

        # ============================================================
        # RIGHT WALL
        # ============================================================

        right = self.create_wall_with_openings(
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
        right = self.apply_box_texture(right, wall_path, scale=1.5)
        scene.add_geometry(
            right,
            node_name="right_wall",
        )

        # ============================================================
        # FLOOR
        # ============================================================

        floor = self.make_box(
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

        roof = self.create_gable_roof(
            house_width=width,
            house_depth=depth,
            wall_height=wall_height,
            roof_height=roof_height,
        )
        roof = self.apply_box_texture(roof, roof_path, scale=1.5)
        scene.add_geometry(
            roof,
            node_name="roof",
        )

        # ============================================================
        # WINDOWS / DOORS
        # ============================================================

        for i, opening in enumerate(front_openings):
            self.add_opening_geometry(
                scene,
                opening,
                "front",
                width,
                depth,
                wall_thickness,
                i
            )

        for i, opening in enumerate(back_openings):
            self.add_opening_geometry(
                scene,
                opening,
                "back",
                width,
                depth,
                wall_thickness,
                i
            )

        for i, opening in enumerate(left_openings):
            self.add_opening_geometry(
                scene,
                opening,
                "left",
                width,
                depth,
                wall_thickness,
                i
            )

        for i, opening in enumerate(right_openings):
            self.add_opening_geometry(
                scene,
                opening,
                "right",
                width,
                depth,
                wall_thickness,
                i
            )

        return scene    
        
if __name__ == "__main__":
    house = HOUSE_GEN(
        width=10.0,
        depth=8.0,
        wall_height=3.0,
        wall_thickness=0.20,
        roof_height=2.0,
    )
    # house_scene = create_four_walled_house(width=5.0, depth=7.0, height=3.5, roof_height=2.0)
    
    # Launch standard interactive 3D pyglet viewport frame
    viewer = SceneViewer(
        house.scene,
        start_loop=False
    )
    
    pyglet.clock.schedule_once(maximize_viewer, 0.1)
    pyglet.app.run()