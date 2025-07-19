use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use crate::core::Engine;
use crate::core::renderer::texture::Texture;

#[derive(Clone, Debug, Eq)]
pub struct Material {
    resource_id: u64,
    texture: Texture
}

impl Material {
    pub fn new(texture: Texture) -> Self {
        let resource_id = Engine::next_resource_id();
        
        Material {
            resource_id,
            texture
        }
    }
    
    pub fn resource_id(&self) -> u64 {
        self.resource_id
    }
    
    pub fn texture(&self) -> &Texture {
        &self.texture
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.resource_id == other.resource_id
    }
}

impl PartialOrd for Material {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.resource_id.partial_cmp(&other.resource_id)
    }
}

impl Ord for Material {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resource_id.cmp(&other.resource_id)
    }
}

impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource_id.hash(state);
        self.texture.hash(state);
    }
}