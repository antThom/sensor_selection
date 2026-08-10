import numpy as np
import trimesh
from PIL import Image
from pyglet.gl import GL_REPEAT, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T

def apply_box_texture(mesh, texture_path, scale=1.0):
    texture = Image.open(texture_path)

    # Get vertex coordinates
    vertices = mesh.vertices

    # Normalize X/Y coordinates into [0, 1]
    x = vertices[:, 0]
    y = vertices[:, 1]

    x_range = max(x.max() - x.min(), 1e-8)
    y_range = max(y.max() - y.min(), 1e-8)

    u = (x - x.min()) / x_range
    v = (y - y.min()) / y_range

    # Scale texture repetition
    u *= scale
    v *= scale

    uv = np.column_stack((u, v))

    # Update your texture creation function like this:
    material = trimesh.visual.texture.SimpleMaterial(
        image=texture,
        kwargs={
            'tex_parameter': {
                GL_TEXTURE_WRAP_S: GL_REPEAT,  # Repeat horizontally
                GL_TEXTURE_WRAP_T: GL_REPEAT   # Repeat vertically
            }
        }
    )

    # Initialize an array of UV coordinates for every vertex in the mesh
    uvs = np.zeros((len(mesh.vertices), 2))
    
    # Track assignments to handle shared vertices across different faces
    counts = np.zeros(len(mesh.vertices))

    # Loop through every face and look at its direction (normal vector)
    for face, normal in zip(mesh.faces, mesh.face_normals):
        # Find which major axis the face is pointing along
        axis = np.argmax(np.abs(normal))
        
        # Pull the 3 coordinates for this face
        face_verts = mesh.vertices[face]
        
        # Calculate UV mapping based on the face orientation
        if axis == 0:    # Face points along X (Left/Right walls) -> Use Y and Z coordinates
            u = face_verts[:, 1] * scale
            v = face_verts[:, 2] * scale
        elif axis == 1:  # Face points along Y (Front/Back walls) -> Use X and Z coordinates
            u = face_verts[:, 0] * scale
            v = face_verts[:, 2] * scale
        else:            # Face points along Z (Floor/Roof cap) -> Use X and Y coordinates
            u = face_verts[:, 0] * scale
            v = face_verts[:, 1] * scale
            
        # Assign calculated coordinates back to the respective vertex indices
        for i, idx in enumerate(face):
            uvs[idx][0] += u[i]
            uvs[idx][1] += v[i]
            counts[idx] += 1

    # Average coordinates for vertices shared between neighboring walls
    # Note: For perfect sharp corners without bleeding, consider calling mesh.unmerge_vertices() first
    counts[counts == 0] = 1
    uvs /= counts[:, None]



    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        material=material
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
    door_mesh.apply_translation([0, -half_d + wall_thickness + 0.01, door_h / 2.0])
    door_mesh.visual.face_colors = [120, 80, 54, 255] # Brown Door

    win_w, win_h = 1.2, 1.2
    glass_left = trimesh.creation.box(extents=[0.05, win_w, win_h])
    glass_left.apply_translation([-half_w + wall_thickness + 0.01, 0, height * 0.6])
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
    roof_mesh.visual.face_colors = [150, 50, 50, 255]

    # Compile all independent panels into the final multi-object scene graph
    return trimesh.Scene([front_wall, back_wall, left_wall, right_wall, door_mesh, glass_left, roof_mesh])
# --------------------------------------------------------
# EXECUTION ROUTINE
# --------------------------------------------------------
if __name__ == "__main__":
    house_scene = create_four_walled_house(width=5.0, depth=7.0, height=3.5, roof_height=2.0)
    
    # Launch standard interactive 3D pyglet viewport frame
    house_scene.show()
