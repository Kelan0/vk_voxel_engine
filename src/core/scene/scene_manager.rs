use crate::application::{Key, Ticker};
use crate::core::{Engine, WorldGenerator};
use anyhow::Result;
use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::Component;
use bevy_ecs::world::{EntityWorldMut, World};
use crate::core::scene::world::VoxelWorld;
use crate::{function_name, profile_scope_fn};

pub struct Scene {
    pub ecs: World,
    pub world: VoxelWorld,
    debug_render_enabled: bool,
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
            world,
            debug_render_enabled: false,
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

    pub fn destroy_entity(&mut self, entity: bevy_ecs::entity::Entity) {
        // let mut commands = self.ecs.commands();
        // commands.entity(entity).despawn();
        // // self.ecs.entity(entity).despawn();
        self.ecs.despawn(entity);
    }

    pub fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);
        
        // let r = self.world.increment_change_tick();

        let mut query = self.ecs.query::<(bevy_ecs::entity::Entity, &mut UpdateComponent)>();

        query.iter(&self.ecs).for_each(|(entity, update_component)| {
            let on_render = &update_component.on_render;
            on_render(entity, ticker, engine);
        });
        
        // for (entity, update_component) in query.iter(&self.world) {
        //     let on_render = &update_component.on_render;
        //     on_render(entity, ticker, engine);
        // }
        
        self.world.update_player_position(engine.render_camera().camera.position());
        
        Ok(())
    }

    pub fn render(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);

        if engine.window.input().key_pressed(Key::F2) {
            self.debug_render_enabled = !self.debug_render_enabled;
        }

        self.world.update(ticker, engine)?;

        if self.debug_render_enabled {
            self.world.draw_debug(engine.scene_renderer.debug_render_context())?;
        }

        Ok(())
    }

    pub fn clear_trackers(&mut self) {
        self.ecs.clear_trackers();
    }

    pub fn draw_gui(&mut self, ticker: &mut Ticker, ctx: &egui::Context) {
        self.world.draw_gui(ticker, ctx);
    }

    pub fn world(&self) -> &VoxelWorld {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut VoxelWorld {
        &mut self.world
    }

    pub fn shutdown(&mut self, engine: &mut Engine) {
        self.world.shutdown(engine);
    }
}
