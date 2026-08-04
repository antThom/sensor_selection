#version 150

// Object: bulk temperature K, emissivity, spatial variation K, variation scale.
uniform vec4 thermal_object;
// Effects: sun-facing gain K, atmosphere enabled, texture variation K, reserved.
uniform vec4 thermal_effects;
// Environment: ambient K, sky K, reflected K, atmospheric K.
uniform vec4 thermal_environment;
// Camera: display minimum K, display maximum K, NETD K, extinction per metre.
uniform vec4 thermal_camera;
// Sun direction toward the sun in world space and daylight strength.
uniform vec4 thermal_sun;
uniform vec4 thermal_camera_position;
uniform float thermal_frame;
uniform float thermal_palette;
uniform float thermal_base_transmission;
uniform sampler2D p3d_Texture0;

in vec3 world_position;
in vec3 world_normal;
in vec2 texture_coordinate;
out vec4 frag_color;

float hash12(vec2 point) {
    vec3 p3 = fract(vec3(point.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float spatial_noise(vec3 point) {
    vec3 cell = floor(point);
    vec3 f = fract(point);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash12(cell.xy + cell.z * 17.0);
    float n100 = hash12(cell.xy + vec2(1.0, 0.0) + cell.z * 17.0);
    float n010 = hash12(cell.xy + vec2(0.0, 1.0) + cell.z * 17.0);
    float n110 = hash12(cell.xy + vec2(1.0, 1.0) + cell.z * 17.0);
    float n001 = hash12(cell.xy + (cell.z + 1.0) * 17.0);
    float n101 = hash12(cell.xy + vec2(1.0, 0.0) + (cell.z + 1.0) * 17.0);
    float n011 = hash12(cell.xy + vec2(0.0, 1.0) + (cell.z + 1.0) * 17.0);
    float n111 = hash12(cell.xy + vec2(1.0, 1.0) + (cell.z + 1.0) * 17.0);
    float lower = mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y);
    float upper = mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y);
    return mix(lower, upper, f.z);
}

vec3 ironbow(float value) {
    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 c1 = vec3(0.149, 0.047, 0.290);
    vec3 c2 = vec3(0.518, 0.086, 0.337);
    vec3 c3 = vec3(0.878, 0.294, 0.137);
    vec3 c4 = vec3(1.0, 0.745, 0.216);
    vec3 c5 = vec3(1.0, 1.0, 0.941);
    if (value < 0.20) return mix(c0, c1, value / 0.20);
    if (value < 0.45) return mix(c1, c2, (value - 0.20) / 0.25);
    if (value < 0.70) return mix(c2, c3, (value - 0.45) / 0.25);
    if (value < 0.88) return mix(c3, c4, (value - 0.70) / 0.18);
    return mix(c4, c5, (value - 0.88) / 0.12);
}

void main() {
    vec3 normal = normalize(world_normal);
    if (length(world_normal) < 0.1) normal = vec3(0.0, 0.0, 1.0);

    float spatial = spatial_noise(world_position * thermal_object.w) * 2.0 - 1.0;
    vec3 surface_texture = texture(p3d_Texture0, texture_coordinate).rgb;
    float texture_luminance = dot(surface_texture, vec3(0.2126, 0.7152, 0.0722));
    // Darker surfaces generally absorb more solar energy.  Texture color is
    // not displayed; luminance only perturbs the physical surface temperature.
    float material_delta = (0.5 - texture_luminance)
        * thermal_effects.z * thermal_sun.w;
    float solar_incidence = max(dot(normal, normalize(thermal_sun.xyz)), 0.0);
    float solar_delta = thermal_effects.x * thermal_sun.w * (solar_incidence - 0.25);
    float surface_temperature = max(
        1.0,
        thermal_object.x + spatial * thermal_object.z + solar_delta + material_delta
    );

    float emissivity = clamp(thermal_object.y, 0.01, 1.0);
    float reflected_radiance = pow(thermal_environment.z, 4.0);
    float surface_radiance = emissivity * pow(surface_temperature, 4.0)
        + (1.0 - emissivity) * reflected_radiance;

    float distance_m = length(world_position - thermal_camera_position.xyz);
    float transmission = thermal_base_transmission
        * exp(-thermal_camera.w * distance_m * thermal_effects.y);
    float detector_radiance = transmission * surface_radiance
        + (1.0 - transmission) * pow(thermal_environment.w, 4.0);
    float apparent_temperature = pow(max(detector_radiance, 1.0), 0.25);

    // NETD approximates random detector noise; a smaller fixed-pattern term
    // prevents a perfectly clean computer-generated image.
    float temporal = floor(thermal_frame * 30.0);
    float detector_noise = (hash12(gl_FragCoord.xy + temporal * 0.071) - 0.5)
        * 3.464 * thermal_camera.z;
    float fixed_pattern = (hash12(gl_FragCoord.xy * 0.37) - 0.5)
        * thermal_camera.z;
    apparent_temperature += detector_noise + fixed_pattern;

    float normalized = clamp(
        (apparent_temperature - thermal_camera.x)
        / max(thermal_camera.y - thermal_camera.x, 0.001),
        0.0,
        1.0
    );
    vec3 color;
    if (thermal_palette < 0.5) color = vec3(normalized);
    else if (thermal_palette < 1.5) color = vec3(1.0 - normalized);
    else color = ironbow(normalized);
    frag_color = vec4(color, 1.0);
}
