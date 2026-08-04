#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec3 world_position;
out vec3 world_normal;
out vec2 texture_coordinate;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
    world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    texture_coordinate = p3d_MultiTexCoord0;
}
