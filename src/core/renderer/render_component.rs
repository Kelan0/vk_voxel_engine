use std::sync::Arc;
use bevy_ecs::component::Component;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::Mesh;

#[derive(Component, Clone)]
pub struct RenderComponent<V: Vertex> {
    pub mesh: Arc<Mesh<V>>,
    pub render_type: RenderType,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RenderType {
    Static,
    Dynamic,
}

impl <V: Vertex> RenderComponent<V> {
    pub fn new(mesh: Arc<Mesh<V>>, render_type: RenderType) -> Self {
        RenderComponent{
            mesh,
            render_type,
        }
    }
}