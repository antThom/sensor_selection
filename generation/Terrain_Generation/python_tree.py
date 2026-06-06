from math import sin, cos, tau
from pathlib import Path

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


def _add_vertex(vwriter, nwriter, twriter, pos, normal, uv):
    vwriter.addData3f(*pos)
    nwriter.addData3f(*normal)
    twriter.addData2f(*uv)


def _make_geom_node(name: str, vdata: GeomVertexData, prim: GeomTriangles) -> NodePath:
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
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
            vwriter, nwriter, twriter,
            (x, y, 0.0),
            Vec3(0, 0, -1),
            (0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a))
        )

    for i in range(sides):
        prim.addVertices(bottom_center, bottom_ring[i + 1], bottom_ring[i])
        prim.closePrimitive()

    # Top cap
    top_center = vdata.getNumRows()
    _add_vertex(vwriter, nwriter, twriter, (0.0, 0.0, height), Vec3(0, 0, 1), (0.5, 0.5))

    top_ring = []
    for i in range(sides + 1):
        a = tau * i / sides
        x, y = radius * cos(a), radius * sin(a)
        idx = vdata.getNumRows()
        top_ring.append(idx)
        _add_vertex(
            vwriter, nwriter, twriter,
            (x, y, height),
            Vec3(0, 0, 1),
            (0.5 + 0.5 * cos(a), 0.5 + 0.5 * sin(a))
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


class LowPolyTree():
    def __init__(
        self,
        trunk_radius=0.15,
        trunk_height=1.2,
        canopy_radius=0.85,
        canopy_height=1.7,
        trunk_sides=6,
        canopy_sides=8,
    ):
        
        self.root = NodePath("low_poly_tree")

        self.trunk = make_low_poly_cylinder(
            radius=trunk_radius,
            height=trunk_height,
            sides=trunk_sides,
        )
        self.canopy = make_low_poly_cone(
            radius=canopy_radius,
            height=canopy_height,
            sides=canopy_sides,
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
        trunk_radius=0.12,
        trunk_height=1.4,
        canopy_radius=0.9,
        canopy_height=1.6,
    )
    tree.set_trunk_color(0.45, 0.25, 0.12, 1.0)
    tree.set_canopy_color(0.15, 0.45, 0.15, 1.0)
    tree.root.flattenStrong()
    filename = Path("generation","Terrain_Generation","terrain_features","quick_tree.bam")
    tree.root.writeBamFile(filename)