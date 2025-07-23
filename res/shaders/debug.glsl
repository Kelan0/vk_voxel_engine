#version 450
#extension GL_EXT_nonuniform_qualifier : enable


struct CameraData {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 viewProjectionMatrix;
};

struct ObjectData {
    mat4 modelMatrix;
    uint colour;
};


vec4 colour_uint_to_vec4(uint colour) {
    return vec4(
        ((colour) & 0xFFu) / 255.0,
        ((colour >> 8) & 0xFFu) / 255.0,
        ((colour >> 16) & 0xFFu) / 255.0,
        ((colour >> 24) & 0xFFu) / 255.0
    );
}


#ifdef VERTEX_SHADER_MODULE
layout(std140, set = 0, binding = 0) uniform UBO1 {
    CameraData camera;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectDataBuffer {
    ObjectData objects[];
};

layout(std140, set = 1, binding = 1) readonly buffer ObjectIndexBuffer {
    uvec4 objectIndices[];
};

layout(location = 0) in vec3 vs_position;
layout(location = 1) in vec3 vs_normal;
layout(location = 2) in vec4 vs_colour;

layout(location = 0) out vec4 fs_colour;

void main() {
    uint objectIndex = objectIndices[gl_InstanceIndex / 4][gl_InstanceIndex % 4];

    mat4 modelMatrix = objects[objectIndex].modelMatrix;
    uint colour = objects[objectIndex].colour;

    fs_colour = vs_colour * vec4(colour_uint_to_vec4(colour));

    gl_Position = camera.viewProjectionMatrix * modelMatrix * vec4(vs_position, 1.0);
}
#endif



#ifdef FRAGMENT_SHADER_MODULE
layout(location = 0) in vec4 fs_colour;


layout(location = 0) out vec4 out_colour;


void main() {
    out_colour = vec4(fs_colour);
}
#endif

