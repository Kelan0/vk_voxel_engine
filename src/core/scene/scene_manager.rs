use crate::application::Ticker;
use crate::core::{Engine, WorldGenerator};
use anyhow::Result;
use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::Component;
use bevy_ecs::world::{EntityWorldMut, World};
use crate::core::scene::world::VoxelWorld;

pub struct Scene {
    pub ecs: World,
    pub world: VoxelWorld,
}

#[derive(Component)]
pub struct EntityNameComponent {
    name: String
}

pub type OnRenderCallback = Box<dyn Fn(bevy_ecs::entity::Entity, &mut Ticker, &mut Engine) + Send + Sync + 'static>;

#[derive(Component)]
pub struct UpdateComponent {
    pub on_render: OnRenderCallback,
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
        let ecs = World::new();
        let world_generator = WorldGenerator::new(WorldGenerator::default_terrain_generator(99));
        let world = VoxelWorld::new(world_generator);
        Ok(Scene {
            ecs,
            world
        })
    }

    pub fn create_entity(&mut self, name: &str) -> Entity {
        let name_component = EntityNameComponent{
            name: name.to_string()
        };

        let entity = self.ecs.spawn(name_component);
        Entity{
            id: entity,
            // scene: self
        }
    }
    
    pub fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        // let r = self.world.increment_change_tick();
        self.ecs.clear_trackers();

        let mut query = self.ecs.query::<(bevy_ecs::entity::Entity, &mut UpdateComponent)>();

        query.iter(&self.ecs).for_each(|(entity, update_component)| {
            let on_render = &update_component.on_render;
            on_render(entity, ticker, engine);
        });
        
        // for (entity, update_component) in query.iter(&self.world) {
        //     let on_render = &update_component.on_render;
        //     on_render(entity, ticker, engine);
        // }
        
        self.world.update_player_position(engine.scene_renderer.camera().position());
        
        Ok(())
    }

    pub fn render(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {

        self.world.update(ticker, engine)?;
        self.world.draw_debug(engine.scene_renderer.debug_render_context())?;

        Ok(())
    }

    pub fn world(&self) -> &VoxelWorld {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut VoxelWorld {
        &mut self.world
    }
}
