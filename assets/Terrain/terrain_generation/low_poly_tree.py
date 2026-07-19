import numpy as np
import os
def save_obj(filename, vertices, faces):
    normals = compute_vertex_normals(vertices, faces)

    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        for face in faces:
            # OBJ format: f v1//n1 v2//n2 v3//n3
            f.write("f " + " ".join(f"{i+1}//{i+1}" for i in face) + "\n")

def create_cylinder(radius, height, segments, z_offset=0):
    angle_step = 2 * np.pi / segments
    vertices = []
    for i in range(segments):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, z_offset])          # bottom circle
        vertices.append([x, y, z_offset + height]) # top circle

    faces = []
    for i in range(segments):
        i0 = i * 2
        i1 = (i * 2 + 2) % (segments * 2)
        i2 = i * 2 + 1
        i3 = (i * 2 + 3) % (segments * 2)
        faces.append([i0, i1, i3, i2])  # quad side

    return vertices, faces

def create_cone(radius, height, segments, z_offset):
    angle_step = 2 * np.pi / segments
    apex = [0, 0, z_offset + height]
    base = [[radius * np.cos(i * angle_step), radius * np.sin(i * angle_step), z_offset] for i in range(segments)]
    vertices = [apex] + base

    faces = []
    for i in range(segments):
        base_idx = i + 1
        next_idx = ((i + 1) % segments) + 1
        faces.append([0, next_idx, base_idx])  # triangle
    return vertices, faces

def compute_vertex_normals(vertices, faces):
    vertex_normals = np.zeros((len(vertices), 3), dtype=np.float32)
    for face in faces:
        v0, v1, v2 = [np.array(vertices[i]) for i in face[:3]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)  # Normalize
        for idx in face:
            vertex_normals[idx] += face_normal
    # Normalize all vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1)
    norms[norms == 0] = 1e-8
    vertex_normals /= norms[:, np.newaxis]
    return vertex_normals

def generate_tree_obj(output_file='lowpoly_tree.obj'):
    all_vertices = []
    all_faces = []

    # Create trunk
    trunk_verts, trunk_faces = create_cylinder(radius=0.2, height=1.5, segments=6)
    all_vertices.extend(trunk_verts)
    all_faces.extend([[v + len(all_vertices) - len(trunk_verts) for v in face] for face in trunk_faces])

    # Create foliage (single cone)
    cone_verts, cone_faces = create_cone(radius=0.8, height=1.5, segments=8, z_offset=1.5)
    all_vertices.extend(cone_verts)
    all_faces.extend([[v + len(all_vertices) - len(cone_verts) for v in face] for face in cone_faces])

    save_obj(output_file, all_vertices, all_faces)
    print(f"✅ Exported low-poly tree to '{output_file}'")

if __name__ == '__main__':
    output_file = r'generation\Terrain_Generation\terrain_features\lowpoly_tree.obj'
    generate_tree_obj(output_file)
