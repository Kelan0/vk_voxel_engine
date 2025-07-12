use bevy_ecs::component::Component;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::Mesh;

#[derive(Component, Clone)]
pub struct RenderComponent<V: Vertex> {
    pub mesh: Mesh<V>
}

impl <V: Vertex> RenderComponent<V> {
    pub fn new(mesh: Mesh<V>) -> Self {
        RenderComponent{
            mesh
        }
    }
}