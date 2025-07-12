use anyhow::Result;
use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::Component;
use bevy_ecs::world::{EntityWorldMut, World};

pub struct Scene {
    pub world: World,
}

#[derive(Component)]
pub struct EntityNameComponent {
    name: String
}

pub struct Entity<'a> {
    id: EntityWorldMut<'a>,
    // scene: &'a mut Scene
}

impl <'a> Entity<'a> {
    pub fn id(&self) -> &EntityWorldMut {
        &self.id
    }
    
    pub fn add_component<C: Bundle>(&mut self, component: C) -> &mut Self {
        self.id.insert(component);
        self
    }
}

impl Scene {
    pub fn new() -> Result<Self> {
        let world = World::new();
        Ok(Scene {
            world
        })
    }

    pub fn create_entity(&mut self, name: &str) -> Entity {
        let name_component = EntityNameComponent{
            name: name.to_string()
        };

        let entity = self.world.spawn(name_component);
        Entity{
            id: entity,
            // scene: self
        }
    }
}
