from __future__ import annotations
import numpy as np
import trimesh
from trimesh.viewer import SceneViewer
from PIL import Image
import ctypes
import pyglet
import yaml
from pathlib import Path
from typing import Dict
from pyglet.gl import GL_REPEAT, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE

def maximize_viewer(dt):
    SW_MAXIMIZE = 3

    hwnd = ctypes.windll.user32.GetForegroundWindow()

    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)

class HOUSE_GEN():
    # Default configuration values - tries first path, falls back to second
    DEFAULT_CONFIG_PATH = Path("assets", "Terrain", "Generate", "house", "config", "extended_house.yaml")
    DEFAULT_HOUSE_PATHS = [
        # ("single_family", "1_story", "basic"),
        ("apartment", "3_story", "walkup"),
    ]
    
    def __init__(self, width=10.0, depth=8.0, wall_height=3.0, wall_thickness=0.20, roof_height=2.0, config_file=None):
        """Initialize house generator with configuration from YAML or parameters."""
        if config_file:
            self.scene = self._load_from_file(config_file)
        else:
            self.scene = self._load_default_or_create(self.DEFAULT_CONFIG_PATH)
    
    def _load_from_file(self, config_file: str) -> trimesh.Scene:
        """Load house configuration from a YAML file."""
        try:
            with open(str(config_file), "r") as file:
                config = yaml.safe_load(file)
                print(f"Loaded custom house configuration from {config_file}")
                return self.create_house(config)
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            raise
    
    def _load_default_or_create(self, config_path: Path) -> trimesh.Scene:
        """Load default house configuration or create empty scene."""
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        try:
            with open(str(config_path), "r") as file:
                config = yaml.safe_load(file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Error loading configuration file: {e}")
            raise

        # Try each default house path in order
        for house_path in self.DEFAULT_HOUSE_PATHS:
            try:
                house = config
                for key in house_path:
                    house = house[key]
                print(f"Loaded default house configuration: {house_path}")
                return self.create_house(house)
            except KeyError:
                continue
        
        # If none of the paths work, show available options
        print(f"Error: Could not find any of the default house configurations in {config_path}")
        print(f"Tried paths: {self.DEFAULT_HOUSE_PATHS}")
        raise KeyError(f"No valid house configuration found. Available paths: {self.DEFAULT_HOUSE_PATHS}")

    def resolve_offset(self, opening, wall_width):
        offset = opening["offset"]

        # Old format
        if isinstance(offset, (int, float)):
            return offset

        reference = offset["reference"]
        margin = offset.get("margin", 0.0)

        if reference == "left":
            return margin

        if reference == "center":
            return wall_width / 2.0

        if reference == "right":
            return wall_width - margin

        raise ValueError(f"Unknown offset reference '{reference}'")
    
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

        def apply_uv_for_axis(axis_index, coord_indices):
            """Apply UV mapping for a specific axis."""
            mask = axes == axis_index
            if not np.any(mask):
                return
                
            fv = face_vertices[mask]
            coords = np.stack([fv[:, :, i] for i in coord_indices], axis=-1)
            
            if style.lower() == "clamp":
                coords_flat = coords.reshape(-1, 2)
                coords_flat = normalize_uv(coords_flat)
                uv = coords_flat.reshape(coords.shape)
            else:
                uv = coords * scale
            
            uvs[faces[mask]] = uv

        # X-facing surfaces: use Y/Z for UV
        apply_uv_for_axis(0, [1, 2])
        
        # Y-facing surfaces: use X/Z for UV
        apply_uv_for_axis(1, [0, 2])
        
        # Z-facing surfaces: use X/Y for UV
        apply_uv_for_axis(2, [0, 1])
            
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
        """Create a door with optional texture."""
        # Door slab
        door_slab = self.make_box(
            (width, thickness, height),
            (0, 0, height / 2),
        )
        
        if texture:
            door_slab = self.apply_box_texture(door_slab, texture, 1.0, "clamp")

        return door_slab



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
        Supports stacked openings (same X, different Z).
        """
        wall_parts = []

        if not openings:
            # No openings, create solid wall
            return self.make_box(
                (length, thickness, height),
                (length / 2, position[0], z_min + height / 2)
            )

        # Resolve and sort openings
        resolved_openings = []
        for opening in openings:
            opening_copy = opening.copy()
            opening_copy["offset"] = self.resolve_offset(opening, length)
            resolved_openings.append(opening_copy)

        # Sort by offset (X position), then by Z height
        resolved_openings.sort(key=lambda o: (o["offset"], o["z"]))
        
        # Group openings by their X position to handle stacking
        opening_groups = {}  # offset -> list of openings at that offset
        for opening in resolved_openings:
            offset = opening["offset"]
            if offset not in opening_groups:
                opening_groups[offset] = []
            opening_groups[offset].append(opening)
        
        # Process each opening group
        current = 0.0
        
        for offset in sorted(opening_groups.keys()):
            openings_at_offset = opening_groups[offset]
            
            # Get the bounds of this opening group
            min_width = min(o["width"] for o in openings_at_offset)
            max_width = max(o["width"] for o in openings_at_offset)
            # Use the first opening's width for the wall segment cutout
            group_width = openings_at_offset[0]["width"]
            
            opening_start = offset - group_width / 2
            opening_end = offset + group_width / 2

            # ----------------------------------------------------------
            # WALL SECTION BEFORE THIS OPENING GROUP
            # ----------------------------------------------------------
            if opening_start > current:
                segment_length = opening_start - current
                wall_parts.append(self._create_wall_segment(
                    axis, position, current, segment_length, height, thickness, z_min
                ))

            # ----------------------------------------------------------
            # PROCESS STACKED OPENINGS AT THIS X POSITION
            # ----------------------------------------------------------
            # Sort by Z height within this group
            openings_at_offset.sort(key=lambda o: o["z"])
            
            current_z = z_min
            for opening in openings_at_offset:
                wz0 = opening["z"]
                wz1 = wz0 + opening["height"]
                
                # Create wall segment BELOW this opening (if there's a gap)
                if wz0 > current_z:
                    gap_height = wz0 - current_z
                    wall_parts.append(self._create_opening_frame_segment(
                        axis, position, offset, group_width, gap_height, thickness, current_z
                    ))
                
                # Update current_z to the top of this opening
                current_z = max(current_z, wz1)
            
            # Create wall segment ABOVE all openings in this group
            if current_z < z_min + height:
                gap_height = z_min + height - current_z
                wall_parts.append(self._create_opening_frame_segment(
                    axis, position, offset, group_width, gap_height, thickness, current_z
                ))

            current = opening_end

        # --------------------------------------------------------------
        # FINAL WALL SECTION (after last opening group)
        # --------------------------------------------------------------
        if current < length:
            segment_length = length - current
            wall_parts.append(self._create_wall_segment(
                axis, position, current, segment_length, height, thickness, z_min
            ))

        return trimesh.util.concatenate(wall_parts) if wall_parts else trimesh.Trimesh()

    def _create_wall_segment(
        self,
        axis: str,
        position: tuple[float, float],
        segment_start: float,
        segment_length: float,
        height: float,
        thickness: float,
        z_min: float,
    ) -> trimesh.Trimesh:
        """Create a wall segment along the specified axis."""
        if axis == "x":
            # Wall runs along X axis
            center = (segment_start + segment_length / 2, position[0], z_min + height / 2)
            size = (segment_length, thickness, height)
            wall_segment = self.make_box(size, center)
        else:  # axis == "y"
            # Wall runs along Y axis
            center = (position[0], segment_start + segment_length / 2, z_min + height / 2)
            size = (thickness, segment_length, height)
            wall_segment = self.make_box(size, center)
      
                
        return wall_segment

    def _create_opening_frame_segment(
        self,
        axis: str,
        position: tuple[float, float],
        offset: float,
        width: float,
        height: float,
        thickness: float,
        z_start: float,
    ) -> trimesh.Trimesh:
        """Create a wall segment around an opening (above/below)."""
        if axis == "x":
            center = (offset, position[0], z_start + height / 2)
            size = (width, thickness, height)
        else:  # axis == "y"
            center = (position[0], offset, z_start + height / 2)
            size = (thickness, width, height)
        
        return self.make_box(size, center)


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

        wall: "front", "back", "left", or "right"
        """
        width = opening["width"]
        height = opening["height"]
        thickness = opening["thickness"]
        z = opening["z"]
        
        # Resolve offset based on wall length
        if wall in ("front", "back"):
            wall_length = house_width
        else:  # "left", "right"
            wall_length = house_depth - 2 * wall_thickness
        
        offset = self.resolve_offset(opening, wall_length)

        if opening["type"] == "window":
            opening_mesh = self.create_window(width, height, thickness)
        elif opening["type"] == "door":
            opening_mesh = self.create_door(width, height, thickness, opening.get("texture", ""))
        else:
            raise ValueError(f"Unknown opening type: {opening['type']}")

        # Position mesh based on wall
        if wall == "front":
            opening_mesh.apply_translation((offset, thickness / 2, z))
        elif wall == "back":
            opening_mesh.apply_translation((offset, house_depth - thickness / 2, z))
        elif wall == "left":
            opening_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
            )
            opening_mesh.apply_translation((thickness / 2, offset + wall_thickness, z))
        elif wall == "right":
            opening_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
            )
            opening_mesh.apply_translation((house_width - thickness / 2, offset + wall_thickness, z))

        scene.add_geometry(opening_mesh, node_name=f"{wall}_{opening['type']}_{num}")


    # --------------------------------------------------------
    # EXECUTION ROUTINE
    # --------------------------------------------------------
    def create_house(
        self,
        house_config: Dict = None,
        wall_path: str = "assets//textures//building_materials//bricks//Bricks097_1K-JPG//Bricks097_1K-JPG_Color.jpg",
        door_path: str = "assets//textures//building_materials//door//wood_panel_door_glass.png",
        roof_path: str = "assets//textures//building_materials//roof//roof_shingles.png"
    ) -> trimesh.Scene:
        scene = trimesh.Scene()
        
        # Extract configuration once to avoid repeated .get() calls
        width = house_config.get("width", 10.0)
        depth = house_config.get("depth", 10.0)
        wall_config = house_config.get("wall", {})
        wall_height = wall_config.get("height", 3.0)
        wall_thickness = wall_config.get("thickness", 0.2)
        left_wall_angle = wall_config.get("left_wall_angle", 0)
        if left_wall_angle != 0.0:
            left_wall_angle = np.pi/2
        right_wall_angle = wall_config.get("right_wall_angle", 0)
        if right_wall_angle != 0.0 and right_wall_angle != "pi/2":
            right_wall_angle = np.pi
        elif right_wall_angle == "pi/2":
            right_wall_angle = np.pi/2
        roof_config = house_config.get("roof", {})
        roof_height = roof_config.get("roof_height", 2.0)
        roof_type = roof_config.get("type", "gable")
        wall_path = wall_config.get("texture", wall_path)

        # ============================================================
        # OPENINGS
        # ============================================================
        front_openings = house_config.get("front_openings", [])
        back_openings  = house_config.get("back_openings", [])
        left_openings  = house_config.get("left_openings", [])
        right_openings = house_config.get("right_openings", [])
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

        front.apply_translation((0, wall_thickness / 2, 0))
        front = self.apply_box_texture(front, wall_path, scale=1.5)
        scene.add_geometry(front, node_name="front_wall")
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

        back.apply_translation((0, depth - wall_thickness / 2, 0))
        back = self.apply_box_texture(back, wall_path, scale=1.5)
        scene.add_geometry(back, node_name="back_wall")

        # ============================================================
        # LEFT WALL
        # ============================================================

        left = self.create_wall_with_openings(
            length=depth - 2 * wall_thickness,
            height=wall_height,
            thickness=wall_thickness,
            openings=left_openings,
            axis="y",
            position=(0, 0),
        )
        left.apply_transform(trimesh.transformations.rotation_matrix(angle=left_wall_angle, direction=[0, 0, 1],point=(0.0, 0.0, 0.0)))
        left.apply_translation((wall_thickness / 2, wall_thickness, 0))
        left = self.apply_box_texture(left, wall_path, scale=1.5)
        
        scene.add_geometry(left, node_name="left_wall")
        # ============================================================
        # RIGHT WALL
        # ============================================================

        right = self.create_wall_with_openings(
            length=depth - 2 * wall_thickness,
            height=wall_height,
            thickness=wall_thickness,
            openings=right_openings,
            axis="y",
            position=(0, 0),
        )
        right.apply_transform(trimesh.transformations.rotation_matrix(angle=right_wall_angle, direction=[0, 0, 1],point=(0.0, 0.0, 0.0)))
        right.apply_translation((width - wall_thickness / 2, wall_thickness, 0))
        right = self.apply_box_texture(right, wall_path, scale=1.5)
        scene.add_geometry(right, node_name="right_wall")

        # ============================================================
        # FLOOR
        # ============================================================

        floor = self.make_box(
            (width, depth, 0.2),
            (width / 2, depth / 2, -0.1),
        )
        scene.add_geometry(floor, node_name="floor")

        # ============================================================
        # ROOF
        # ============================================================
        if roof_type == "gable":
            roof = self.create_gable_roof(
                house_width=width,
                house_depth=depth,
                wall_height=wall_height,
                roof_height=roof_height,
            )
            roof = self.apply_box_texture(roof, roof_path, scale=1.5)
        else:
            roof = self.make_box(
                (width, depth, roof_height),
                (width / 2, depth / 2, wall_height + roof_height / 2),
            )
            roof = self.apply_box_texture(roof, roof_path, scale=1.5)
        
        scene.add_geometry(roof, node_name="roof")

        # ============================================================
        # WINDOWS / DOORS
        # ============================================================

        # Add opening geometries for all walls
        walls_and_openings = [
            ("front", front_openings),
            ("back", back_openings),
            ("left", left_openings),
            ("right", right_openings),
        ]
        
        for wall_name, openings in walls_and_openings:
            for i, opening in enumerate(openings):
                self.add_opening_geometry(
                    scene,
                    opening,
                    wall_name,
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