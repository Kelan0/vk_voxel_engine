use std::sync::Arc;
use bevy_ecs::component::Component;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::{Material, Mesh};

#[derive(Component, Clone)]
pub struct RenderComponent<V: Vertex> {
    pub render_type: RenderType,
    pub mesh: Option<Arc<Mesh<V>>>,
    pub material: Option<Material>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RenderType {
    Static,
    Dynamic,
}

impl <V: Vertex> RenderComponent<V> {
    pub fn new(render_type: RenderType, mesh: Option<Arc<Mesh<V>>>) -> Self {
        RenderComponent{
            mesh,
            render_type,
            material: None
        }
    }
    
    pub fn render_type(&self) -> RenderType {
        self.render_type
    }
    
    pub fn mesh(&self) -> Option<&Arc<Mesh<V>>> {
        self.mesh.as_ref()
    }
    
    pub fn material(&self) -> Option<&Material> {
        self.material.as_ref()
    }

    pub fn with_material(mut self, material: Option<Material>) -> Self {
        self.material = material;
        self
    }
}