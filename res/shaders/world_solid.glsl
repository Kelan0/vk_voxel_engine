#version 450
#extension GL_EXT_nonuniform_qualifier : enable


struct CameraData {
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 viewProjectionMatrix;
};

struct ObjectData {
    mat4 modelMatrix;
    uint materialIndex;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

//struct Material {
//    uint albedoTextureIndex;
//    uint roughnessTextureIndex;
//    uint metallicTextureIndex;
//    uint emissionTextureIndex;
//    uint normalTextureIndex;
//    uint packedAlbedoColour;
//    uint packedRoughnessMetallicEmissionR;
//    uint packedEmissionGB;
//    uint flags;
//    uint _pad0;
//    uint _pad2;
//    uint _pad3;
//};

struct Material {
    uint textureIndex;
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

layout(location = 0) in vec3 vs_position;
layout(location = 1) in vec3 vs_normal;
layout(location = 2) in vec4 vs_colour;
layout(location = 3) in vec2 vs_texture;

layout(location = 0) out vec4 fs_colour;
layout(location = 1) out vec2 fs_texture;
layout(location = 6) out flat uint fs_materialIndex;

void main() {
    uint objectIndex = objectIndices[gl_InstanceIndex / 4][gl_InstanceIndex % 4];

    mat4 modelMatrix = objects[objectIndex].modelMatrix;
    uint materialIndex = objects[objectIndex].materialIndex;

    fs_colour = vs_colour;
    fs_texture = vs_texture;
    fs_materialIndex = materialIndex;

    gl_Position = camera.viewProjectionMatrix * modelMatrix * vec4(vs_position, 1.0);
}
#endif



#ifdef FRAGMENT_SHADER_MODULE
layout(location = 0) in vec4 fs_colour;
layout(location = 1) in vec2 fs_texture;
layout(location = 6) in flat uint fs_materialIndex;


layout(location = 0) out vec4 out_colour;

layout(set = 2, binding = 0) uniform sampler2D textures[256];

layout(set = 2, binding = 1) readonly buffer MaterialDataBuffer {
    Material objectMaterials[];
};


void main() {

#ifdef WIREFRAME_ENABLED
    out_colour = vec4(1.0, 1.0, 1.0, 1.0);
#else

    Material material = objectMaterials[fs_materialIndex];

    vec3 texColour = texture(textures[material.textureIndex], fs_texture).rgb;

    out_colour = vec4(texColour, 1.0);
#endif

}
#endif

