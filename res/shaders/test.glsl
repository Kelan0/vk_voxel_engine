#version 450


struct CameraData {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 viewProjectionMatrix;
};

#ifdef VERTEX_SHADER_MODULE

layout(std140, set = 0, binding = 0) uniform UBO1 {
    CameraData camera;
};

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 colour;

layout(location = 0) out vec3 fs_colour;

void main() {
    fs_colour = vec3(colour);

    gl_Position = camera.viewProjectionMatrix * vec4(position, 0.0, 1.0);
}
#endif



#ifdef FRAGMENT_SHADER_MODULE
layout(location = 0) in vec3 fs_colour;

layout(location = 0) out vec4 out_color;

void main() {

#ifdef WIREFRAME_ENABLED
    out_color = vec4(1.0, 1.0, 1.0, 1.0);
#else
    out_color = vec4(fs_colour, 1.0);
#endif

}
#endif

