use std::sync::Arc;
use bevy_ecs::component::Component;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::{Material, Mesh};

#[derive(Component, Clone)]
pub struct RenderComponent<V: Vertex> {
    pub render_type: RenderType,
    pub mesh: Arc<Mesh<V>>,
    pub material: Option<Material>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RenderType {
    Static,
    Dynamic,
}

impl <V: Vertex> RenderComponent<V> {
    pub fn new(render_type: RenderType, mesh: Arc<Mesh<V>>) -> Self {
        RenderComponent{
            mesh,
            render_type,
            material: None
        }
    }
    
    pub fn render_type(&self) -> RenderType {
        self.render_type
    }
    
    pub fn mesh(&self) -> &Arc<Mesh<V>> {
        &self.mesh
    }
    
    pub fn material(&self) -> &Option<Material> {
        &self.material
    }

    pub fn with_material(mut self, material: Option<Material>) -> Self {
        self.material = material;
        self
    }
}