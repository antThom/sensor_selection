from math import sin, cos, pi, tau, atan2, acos
from pathlib import Path
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    Texture,
    TextureStage,
    Vec3,
)
import random


def _add_vertex(vwriter, nwriter, twriter, pos, normal, uv):
    vwriter.addData3(*pos)
    nwriter.addData3(*normal)
    twriter.addData2(*uv)


def _make_geom_node(name: str, vdata: GeomVertexData, prim: GeomTriangles) -> NodePath:
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_blob_lobe_z_up(
    radius=0.9,
    subdivisions=1,
    noise_amplitude=0.25,
) -> NodePath:
    """
    Create a low-poly 'blob' foliage lobe:
      - Start from an octahedron (6 vertices, 8 faces)
      - Optionally subdivide faces
      - Perturb vertices with radial noise
    Z is up.
    """

    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("blob_lobe", fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)

    def add_vertex(pos: Vec3):
        """Adds a vertex and returns its index."""
        idx = vdata.getNumRows()
        vwriter.addData3(pos)
        # Normal and UV will be filled later
        nwriter.addData3(0, 0, 0)
        twriter.addData2(0, 0)
        return idx

    # --- 1) Base octahedron (Z-up) ---

    # Positions of base octahedron (radius 1)
    base_vertices = [
        Vec3(1, 0, 0),  # 0
        Vec3(-1, 0, 0),  # 1
        Vec3(0, 1, 0),  # 2
        Vec3(0, -1, 0),  # 3
        Vec3(0, 0, 1),  # 4 top
        Vec3(0, 0, -1),  # 5 bottom
    ]

    # Faces (triangles) indexing into base_vertices
    base_faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]

    # --- 2) Subdivide faces (optional) ---

    vertices = base_vertices[:]
    faces = list(base_faces)

    def midpoint(a: Vec3, b: Vec3) -> Vec3:
        return (a + b) * 0.5

    for _ in range(subdivisions):
        new_faces = []
        mid_cache = {}

        def get_mid_index(i0, i1):
            key = tuple(sorted((i0, i1)))
            if key in mid_cache:
                return mid_cache[key]
            p = midpoint(vertices[i0], vertices[i1])
            # Normalize to roughly spherical shape
            if p.length_squared() > 0:
                p.normalize()
            idx = len(vertices)
            vertices.append(p)
            mid_cache[key] = idx
            return idx

        for f in faces:
            i0, i1, i2 = f
            m01 = get_mid_index(i0, i1)
            m12 = get_mid_index(i1, i2)
            m20 = get_mid_index(i2, i0)

            # 4 subfaces
            new_faces.append((i0, m01, m20))
            new_faces.append((i1, m12, m01))
            new_faces.append((i2, m20, m12))
            new_faces.append((m01, m12, m20))

        faces = new_faces

    # --- 3) Apply radial noise and scale to target radius ---

    for i, p in enumerate(vertices):
        if p.length_squared() > 0:
            p.normalize()
        # Random radial noise
        noise = random.uniform(-noise_amplitude, noise_amplitude)
        p *= radius + noise
        vertices[i] = p

    # --- 4) Add vertices to GeomVertexData ---

    index_map = []
    for p in vertices:
        idx = add_vertex(p)
        index_map.append(idx)

    # --- 5) Build triangle primitives and accumulate face normals ---

    # We first build triangles, but also accumulate normals per vertex,
    # then normalize them afterwards for smooth shading.
    # For a harder, faceted look, you’d duplicate vertices per face instead.

    # Temporary normal accumulator
    accum_normals = [Vec3(0, 0, 0) for _ in vertices]

    for f in faces:
        i0, i1, i2 = f
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        e1 = v1 - v0
        e2 = v2 - v0
        n = e1.cross(e2)
        if n.length_squared() != 0:
            n.normalize()
        else:
            n = Vec3(0, 0, 1)

        accum_normals[i0] += n
        accum_normals[i1] += n
        accum_normals[i2] += n

        prim.addVertices(index_map[i0], index_map[i1], index_map[i2])
        prim.closePrimitive()

    # --- 6) Write averaged normals and simple spherical UVs ---

    # Reset writers to overwrite normal/uv columns
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")

    for i, p in enumerate(vertices):
        n = accum_normals[i]
        if n.length_squared() != 0:
            n.normalize()
        else:
            # fallback to radial normal
            n = p.normalized() if p.length_squared() != 0 else Vec3(0, 0, 1)

        # Simple spherical UV projection
        # theta = angle around Z, phi = angle from Z
        theta = (pi + atan2(p.y, p.x)) / (2 * pi)  # [0,1]
        # for phi we can do acos(z / r)
        r = p.length()
        z = p.z / (r if r != 0 else 1)
        z = max(-1.0, min(1.0, z))
        phi = acos(z) / pi  # [0,1]

        nwriter.setData3(n)
        twriter.setData2(theta, phi)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("blob_lobe_geom")
    node.addGeom(geom)
    return NodePath(node)


def make_low_poly_cylinder(radius=0.15, height=1.2, sides=6) -> NodePath:
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("trunk", fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)

    # Side faces
    for i in range(sides):
        a0 = tau * i / sides
        a1 = tau * (i + 1) / sides

        x0, y0 = radius * cos(a0), radius * sin(a0)
        x1, y1 = radius * cos(a1), radius * sin(a1)

        # Flat face normal for the side
        mid = (a0 + a1) * 0.5
        normal = Vec3(cos(mid), sin(mid), 0)

        base = vdata.getNumRows()

        _add_vertex(vwriter, nwriter, twriter, (x0, y0, 0.0), normal, (0.0, 0.0))
        _add_vertex(vwriter, nwriter, twriter, (x1, y1, 0.0), normal, (1.0, 0.0))
        _add_vertex(vwriter, nwriter, twriter, (x1, y1, height), normal, (1.0, 1.0))
        _add_vertex(vwriter, nwriter, twriter, (x0, y0, height), normal, (0.0, 1.0))

        prim.addVertices(base, base + 1, base + 2)
        prim.closePrimitive()
        prim.addVertices(base, base + 2, base + 3)
        prim.closePrimitive()

    # Bottom cap
    bottom_center = vdata.getNumRows()
    _add_vertex(vwriter, nwriter, twriter, (0.0, 0.0, 0.0), Vec3(0, 0, -1), (0.5, 0.5))

    bottom_ring = []
    for i in range(sides + 1):
        a = tau * i / sides
        x, y = radius * cos(a), radius * sin(a)
        idx = vdata.getNumRows()
        bottom_ring.append(idx)
        _add_vertex(
            vwriter,
            nwriter,
            twriter,
            (x, y, 0.0),
            Vec3(0, 0, -1),
            (0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a)),
        )

    for i in range(sides):
        prim.addVertices(bottom_center, bottom_ring[i + 1], bottom_ring[i])
        prim.closePrimitive()

    # Top cap
    top_center = vdata.getNumRows()
    _add_vertex(
        vwriter, nwriter, twriter, (0.0, 0.0, height), Vec3(0, 0, 1), (0.5, 0.5)
    )

    top_ring = []
    for i in range(sides + 1):
        a = tau * i / sides
        x, y = radius * cos(a), radius * sin(a)
        idx = vdata.getNumRows()
        top_ring.append(idx)
        _add_vertex(
            vwriter,
            nwriter,
            twriter,
            (x, y, height),
            Vec3(0, 0, 1),
            (0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a)),
        )

    for i in range(sides):
        prim.addVertices(top_center, top_ring[i], top_ring[i + 1])
        prim.closePrimitive()

    return _make_geom_node("trunk_geom", vdata, prim)


def make_low_poly_cone(radius=0.85, height=1.7, sides=8) -> NodePath:
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("canopy", fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)

    apex = (0.0, 0.0, height)

    # One triangle per face for a low-poly look.
    # UVs are simple and work well enough for a diffuse texture.
    for i in range(sides):
        a0 = tau * i / sides
        a1 = tau * (i + 1) / sides

        p0 = (radius * cos(a0), radius * sin(a0), 0.0)
        p1 = (radius * cos(a1), radius * sin(a1), 0.0)

        # Compute a face normal
        v0 = Vec3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v1 = Vec3(apex[0] - p0[0], apex[1] - p0[1], apex[2] - p0[2])
        normal = v0.cross(v1)
        normal.normalize()

        base = vdata.getNumRows()

        _add_vertex(vwriter, nwriter, twriter, p0, normal, (0.0, 0.0))
        _add_vertex(vwriter, nwriter, twriter, p1, normal, (1.0, 0.0))
        _add_vertex(vwriter, nwriter, twriter, apex, normal, (0.5, 1.0))

        prim.addVertices(base, base + 1, base + 2)
        prim.closePrimitive()

    return _make_geom_node("canopy_geom", vdata, prim)


def make_realistic_trunk(
    radius_bottom=0.15,
    radius_top=0.08,
    height=1.5,
    sides=8,
    bend_strength=0.1,
) -> NodePath:
    """
    Low-poly trunk:
    - Slight taper (radius_bottom -> radius_top)
    - Slight bend in X using a quadratic curve
    """
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("trunk", fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)

    def _add_vertex(pos, normal, uv):
        vwriter.addData3(*pos)
        nwriter.addData3(*normal)
        twriter.addData2(*uv)

    # param t in [0,1] along height
    def trunk_center_at(t):
        # z is height, x has a small quadratic bend
        z = t * height
        x = bend_strength * (t**2)  # bend towards +X
        y = 0.0
        return Vec3(x, y, z)

    # Build rings at bottom and top
    bottom_ring = []
    top_ring = []

    center_bottom = trunk_center_at(0.0)
    center_top = trunk_center_at(1.0)

    for i in range(sides):
        a = tau * i / sides
        # tangent around trunk local radial direction
        dx = cos(a)
        dy = sin(a)

        # bottom ring
        p_bottom = center_bottom + Vec3(dx * radius_bottom, dy * radius_bottom, 0.0)
        # top ring
        p_top = center_top + Vec3(dx * radius_top, dy * radius_top, 0.0)

        bottom_ring.append(p_bottom)
        top_ring.append(p_top)

    # Create side faces
    for i in range(sides):
        i1 = (i + 1) % sides

        p0 = bottom_ring[i]
        p1 = bottom_ring[i1]
        p2 = top_ring[i1]
        p3 = top_ring[i]

        # Face normal: (p1-p0) x (p3-p0)
        v0 = p1 - p0
        v1 = p3 - p0
        normal = v0.cross(v1)
        if normal.length_squared() != 0:
            normal.normalize()
        else:
            normal = Vec3(0, 0, 1)

        base = vdata.getNumRows()

        # simple cylindrical UVs: u = angle, v = height
        u0 = i / sides
        u1 = (i + 1) / sides

        _add_vertex(p0, normal, (u0, 0.0))
        _add_vertex(p1, normal, (u1, 0.0))
        _add_vertex(p2, normal, (u1, 1.0))
        _add_vertex(p3, normal, (u0, 1.0))

        prim.addVertices(base, base + 1, base + 2)
        prim.addVertices(base, base + 2, base + 3)
        prim.closePrimitive()
        prim.closePrimitive()

    # Bottom cap (optional)
    center_idx = vdata.getNumRows()
    _add_vertex(center_bottom, Vec3(0, 0, -1), (0.5, 0.5))
    ring_indices = []
    for i in range(sides + 1):
        a = tau * i / sides
        dx, dy = cos(a), sin(a)
        p = center_bottom + Vec3(dx * radius_bottom, dy * radius_bottom, 0.0)
        idx = vdata.getNumRows()
        ring_indices.append(idx)
        _add_vertex(p, Vec3(0, 0, -1), (0.5 + 0.5 * dx, 0.5 + 0.5 * dy))

    for i in range(sides):
        prim.addVertices(center_idx, ring_indices[i + 1], ring_indices[i])
        prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("realistic_trunk")
    node.addGeom(geom)
    return NodePath(node)


def make_canopy_lobe(radius=0.9, height=1.0, sides=8) -> NodePath:
    """
    Single low-poly cone, like your original, but we’ll use several of these as lobes.
    """
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("canopy_lobe", fmt, Geom.UHStatic)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")
    prim = GeomTriangles(Geom.UHStatic)

    def _add_vertex(pos, normal, uv):
        vwriter.addData3(*pos)
        nwriter.addData3(*normal)
        twriter.addData2(*uv)

    apex = (0.0, 0.0, height)

    for i in range(sides):
        a0 = tau * i / sides
        a1 = tau * (i + 1) / sides

        p0 = (radius * cos(a0), radius * sin(a0), 0.0)
        p1 = (radius * cos(a1), radius * sin(a1), 0.0)

        # face normal
        v0 = Vec3(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v1 = Vec3(apex[0] - p0[0], apex[1] - p0[1], apex[2] - p0[2])
        normal = v0.cross(v1)
        if normal.length_squared() != 0:
            normal.normalize()
        else:
            normal = Vec3(0, 1, 0)

        base = vdata.getNumRows()

        _add_vertex(p0, normal, (0.0, 0.0))
        _add_vertex(p1, normal, (1.0, 0.0))
        _add_vertex(apex, normal, (0.5, 1.0))

        prim.addVertices(base, base + 1, base + 2)
        prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("canopy_lobe_geom")
    node.addGeom(geom)
    return NodePath(node)


def make_clustered_canopy(
    base_radius=0.9,
    base_height=1.2,
    lobes=4,
    sides=8,
    spread=0.4,
    vertical_spread=0.3,
) -> NodePath:
    """
    Build a canopy made of multiple overlapping cone lobes
    to look more organic.
    """
    root = NodePath("clustered_canopy")

    for i in range(lobes):
        lobe = make_canopy_lobe(radius=base_radius, height=base_height, sides=sides)

        # Random horizontal offset
        angle = random.random() * tau
        dist = random.random() * spread
        dx = cos(angle) * dist
        dz = sin(angle) * dist

        # Slight vertical variation
        dy = random.uniform(-vertical_spread, vertical_spread)

        # Random scale
        scale = random.uniform(0.8, 1.2)
        lobe.setScale(scale)

        # Random rotation
        lobe.setH(random.uniform(0, 360))

        # Position lobe
        lobe.setPos(
            dx, dy, dz
        )  # note: trunk's Y is height; here Z is sideways if trunk uses Y-up

        lobe.reparentTo(root)

    return root


def make_clustered_blob_canopy_z_up(
    base_radius=0.9,
    base_height=1.4,
    lobes=4,
    subdivisions=1,
    noise_amplitude=0.25,
    horizontal_spread=0.5,
    vertical_spread=0.3,
) -> NodePath:
    """
    Builds a canopy made of several overlapping blob lobes.
    Z is up; canopy origin is roughly its center.
    """
    root = NodePath("clustered_blob_canopy")

    for _ in range(lobes):
        lobe = make_blob_lobe_z_up(
            radius=base_radius,
            subdivisions=subdivisions,
            noise_amplitude=noise_amplitude,
        )

        # Random horizontal offset (X/Y plane)
        angle = random.random() * tau
        dist = random.random() * horizontal_spread
        dx = cos(angle) * dist
        dy = sin(angle) * dist

        # Vertical variation
        dz = random.uniform(-vertical_spread, vertical_spread)

        # Random scale
        scale = random.uniform(0.8, 1.2)
        lobe.setScale(scale)

        # Random slight rotation around Z
        lobe.setH(random.uniform(0, 360))

        lobe.setPos(dx, dy, base_height * 0.5 + dz)
        lobe.reparentTo(root)

    return root


class LowPolyTree:
    def __init__(
        self,
        trunk_radius_bottom=0.15,
        trunk_radius_top=0.08,
        trunk_height=1.5,
        trunk_sides=8,
        canopy_radius=0.9,
        canopy_height=1.4,
        canopy_lobes=4,
        canopy_subdivisions=1,
    ):
        self.root = NodePath("low_poly_tree")

        self.trunk = make_realistic_trunk(
            radius_bottom=trunk_radius_bottom,
            radius_top=trunk_radius_top,
            height=trunk_height,
            sides=trunk_sides,
            bend_strength=0.1,
        )

        self.canopy = make_clustered_blob_canopy_z_up(
            base_radius=canopy_radius,
            base_height=canopy_height,
            lobes=canopy_lobes,
            subdivisions=canopy_subdivisions,
            noise_amplitude=0.25,
            horizontal_spread=0.4,
            vertical_spread=0.2,
        )

        self.trunk.reparentTo(self.root)
        self.canopy.reparentTo(self.root)
        self.canopy.setZ(trunk_height)

        # Keeps the model visible from both sides if winding/culling becomes annoying.
        # Remove this if you prefer strict back-face culling.
        self.root.setTwoSided(True)

    def set_trunk_color(self, r, g, b, a=1.0):
        self.trunk.setColor(r, g, b, a)

    def set_canopy_color(self, r, g, b, a=1.0):
        self.canopy.setColor(r, g, b, a)

    def set_trunk_texture(self, loader, texture_path, tex_scale=(1, 1)):
        tex = loader.loadTexture(str(texture_path))
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_repeat)
        self.trunk.setTexture(tex, 1)
        self.trunk.setTexScale(TextureStage.getDefault(), tex_scale[0], tex_scale[1])

    def set_canopy_texture(self, loader, texture_path, tex_scale=(1, 1)):
        tex = loader.loadTexture(str(texture_path))
        tex.setWrapU(Texture.WM_repeat)
        tex.setWrapV(Texture.WM_repeat)
        self.canopy.setTexture(tex, 1)
        self.canopy.setTexScale(TextureStage.getDefault(), tex_scale[0], tex_scale[1])

    def set_pos(self, x, y, z):
        self.root.setPos(x, y, z)

    def set_scale(self, scale):
        self.root.setScale(scale)

    def attach_to(self, parent):
        self.root.reparentTo(parent)
        return self.root


class VISUAL(ShowBase):
    def __init__(self, tree_np: NodePath):
        ShowBase.__init__(self)
        self.scene = tree_np
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setPos(0, 30, 0)  # move forward a bit to see it


# Example usage inside your Panda3D app:
#
# tree = LowPolyTree(
#     trunk_radius=0.12,
#     trunk_height=1.4,
#     canopy_radius=0.9,
#     canopy_height=1.6,
# )
# tree.attach_to(render)
# tree.set_pos(5, 10, 0)
# tree.set_trunk_color(0.45, 0.25, 0.12, 1.0)
# tree.set_canopy_color(0.15, 0.45, 0.15, 1.0)
# tree.set_trunk_texture(loader, "textures/bark.png", tex_scale=(2, 4))
# tree.set_canopy_texture(loader, "textures/leaves.png", tex_scale=(3, 3))

if __name__ == "__main__":
    tree = LowPolyTree(
        trunk_radius_bottom=1,
        trunk_radius_top=0.5,
        trunk_height=10.8,
        canopy_radius=5.0,
        canopy_height=8.3,
        canopy_lobes=50,
        canopy_subdivisions=10,
    )
    tree.set_trunk_color(0.45, 0.25, 0.12, 1.0)
    tree.set_canopy_color(0.15, 0.45, 0.15, 1.0)
    tree.root.flattenStrong()
    app = VISUAL(tree.root)
    app.run()

    # tree.root.flattenStrong()
    # filename = Path("generation","Terrain_Generation","terrain_features","quick_tree.bam")
    # tree.root.writeBamFile(filename)
