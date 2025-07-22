#version 450
#extension GL_EXT_nonuniform_qualifier : enable


struct CameraData {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 viewProjectionMatrix;
};

struct ObjectData {
    mat4 modelMatrix;
};



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

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

layout(location = 0) out vec3 fs_colour;

void main() {
    uint objectIndex = objectIndices[gl_InstanceIndex / 4][gl_InstanceIndex % 4];

    mat4 modelMatrix = objects[objectIndex].modelMatrix;
    uint materialIndex = objects[objectIndex].materialIndex;

    fs_colour = vec3(normalize(normal) * 0.5 + 0.5);
    fs_materialIndex = materialIndex;

    gl_Position = camera.viewProjectionMatrix * modelMatrix * vec4(position, 1.0);
}
#endif



#ifdef FRAGMENT_SHADER_MODULE
layout(location = 0) in vec3 fs_colour;


layout(location = 0) out vec4 out_color;


void main() {

    Material material = objectMaterials[fs_materialIndex];

    out_color = vec4(fs_colour, 1.0);

}
#endif

