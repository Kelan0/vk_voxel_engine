mod application;
mod core;
mod util;

use crate::application::ticker::TickProfileStatistics;
use crate::application::window::WindowResizedEvent;
use crate::application::Key;
use crate::core::{AxisDirection, BaseVertex, GraphicsManager, Material, Mesh, MeshConfiguration, MeshData, PrimaryCommandBuffer, RecreateSwapchainEvent, RenderComponent, RenderType, Scene, StandardMemoryAllocator, Texture, TextureAtlas, Transform, UpdateComponent, WireframeMode};
use anyhow::Result;
use application::ticker::Ticker;
use application::App;
use bevy_ecs::entity::Entity;
use core::Engine;
use glam::Vec3;
use log::{debug, error, info};
use sdl3::mouse::MouseButton;
use shrev::ReaderId;
use std::fs;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::DeviceSize;
use vulkano::format::Format;
use vulkano::image::ImageUsage;

struct TestGame {
    camera_pitch: f32,
    camera_yaw: f32,
    move_speed: f32,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_window_resized: Option<ReaderId<WindowResizedEvent>>,
    debug_stats: Vec<TickProfileStatistics>,

    test_mesh: Option<Arc<Mesh<BaseVertex>>>
}

impl TestGame {
    fn new() -> Self {
        TestGame {
            camera_pitch: 0.0,
            camera_yaw: 0.0,
            move_speed: 3.0,
            event_recreate_swapchain: None,
            event_window_resized: None,
            debug_stats: vec![],
            test_mesh: None,
        }
    }

    fn on_recreate_swapchain(&mut self, _engine: &Engine) -> Result<()> {
        Ok(())
    }

    fn create_test_mesh(&mut self, allocator: Arc<StandardMemoryAllocator>) -> Result<()>{

        let vertices = [
            BaseVertex { position: [-0.5, 0.5, 0.0], normal: [0.0, 0.0, 1.0], colour: [0.0, 1.0, 1.0], texture: [0.0, 1.0] },
            BaseVertex { position: [0.5, 0.5, 0.0], normal: [0.0, 0.0, 1.0], colour: [1.0, 1.0, 1.0], texture: [1.0, 1.0] },
            BaseVertex { position: [-0.5, -0.5, 0.0], normal: [0.0, 0.0, 1.0], colour: [0.0, 0.0, 1.0], texture: [0.0, 0.0] },
            BaseVertex { position: [0.5, -0.5, 0.0], normal: [0.0, 0.0, 1.0], colour: [1.0, 0.0, 1.0], texture: [1.0, 0.0] },
        ];

        let indices = [0, 1, 2, 1, 3, 2];

        let mesh2 = Arc::new(Mesh::new(allocator.clone(), MeshConfiguration {
            vertices: Vec::from(vertices),
            indices: Some(Vec::from(indices)),
        })?);

        self.test_mesh = Some(mesh2);
        Ok(())
    }

    fn add_test_entity(&self, scene: &mut Scene, pos: Vec3) {

        let mesh2 = self.test_mesh.as_ref().unwrap();

        let render_component = RenderComponent::new(RenderType::Dynamic, mesh2.clone());

        scene.create_entity("TestEntity2")
            .add_component(render_component)
            .add_component(*Transform::new()
                .translate(pos)
                .rotate_local_z(f32::to_radians(30.0)))
            .add_component(UpdateComponent{
                on_render: Box::new(|entity: Entity, ticker: &mut Ticker, engine: &mut Engine| {
                    let mut entity = engine.scene.world.entity_mut(entity);
                    entity.modify_component(|transform: &mut Transform| {
                        transform.rotate_local_y(f64::to_radians(30.0 * ticker.delta_time()) as f32);
                    });
                })
            });
    }
}

impl App for TestGame {
    fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(
            engine
                .graphics
                .event_bus()
                .register::<RecreateSwapchainEvent>(),
        );
        self.event_window_resized =
            Some(engine.window.event_bus().register::<WindowResizedEvent>());
        Ok(())
    }

    fn init(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        info!("Init TestGame");
        let window = &mut engine.window;
        ticker.set_desired_tick_rate(0.0);
        window.set_visible(true);

        let allocator = engine.graphics.memory_allocator();
        
        let buffer = GraphicsManager::create_staging_subbuffer::<u8>(allocator.clone(), 512 * 512 * 4)?;

        let mut cmd_buf = engine.graphics.begin_transfer_commands()?;

        let mut texture_atlas = TextureAtlas::new(allocator.clone(), 16, 4, 4, Format::R8G8B8A8_UNORM, ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST)?;

        texture_atlas.begin_loading(allocator.clone(), Some(buffer.clone()))?;
        texture_atlas.load_texture_from_file(&mut cmd_buf, "crafting_table_top", 0, 0, "res/textures/blocks/crafting_table_top.png")?;
        texture_atlas.load_texture_from_file(&mut cmd_buf, "crafting_table_side", 1, 0, "res/textures/blocks/crafting_table_side.png")?;
        texture_atlas.load_texture_from_file(&mut cmd_buf, "crafting_table_front", 0, 1, "res/textures/blocks/crafting_table_front.png")?;
        texture_atlas.load_texture_from_file(&mut cmd_buf, "planks_oak", 2, 3, "res/textures/blocks/planks_oak.png")?;
        texture_atlas.finish_loading();

        let c0 = texture_atlas.find_coords_for_cell("crafting_table_top").expect("crafting_table_top - Coords not found in texture atles");
        let c1 = texture_atlas.find_coords_for_cell("crafting_table_side").expect("crafting_table_side - Coords not found in texture atles");
        let c2 = texture_atlas.find_coords_for_cell("crafting_table_front").expect("crafting_table_front - Coords not found in texture atles");
        let c3 = texture_atlas.find_coords_for_cell("planks_oak").expect("planks_oak - Coords not found in texture atles");

        let mut mesh_data: MeshData<BaseVertex> = MeshData::new();
        // mesh_data.create_cuboid_textured([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], [0.0, 0.0], [1.0, 1.0]);
        mesh_data.create_box_face_textured(AxisDirection::NegX, [-0.5, -0.5], [0.5, 0.5], 0.5, c1[0], c1[1]);
        mesh_data.create_box_face_textured(AxisDirection::PosX, [-0.5, -0.5], [0.5, 0.5], 0.5, c1[0], c1[1]);
        mesh_data.create_box_face_textured(AxisDirection::NegY, [-0.5, -0.5], [0.5, 0.5], 0.5, c3[0], c3[1]);
        mesh_data.create_box_face_textured(AxisDirection::PosY, [-0.5, -0.5], [0.5, 0.5], 0.5, c0[0], c0[1]);
        mesh_data.create_box_face_textured(AxisDirection::NegZ, [-0.5, -0.5], [0.5, 0.5], 0.5, c1[0], c1[1]);
        mesh_data.create_box_face_textured(AxisDirection::PosZ, [-0.5, -0.5], [0.5, 0.5], 0.5, c2[0], c2[1]);

        let mesh1 = Arc::new(mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf)?);

        engine.graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;


        let render_component_1 = RenderComponent::new(RenderType::Static, mesh1)
            .with_material(Some(Material::new(texture_atlas.texture().clone())));
        
        self.create_test_mesh(allocator.clone())?;

        // engine.scene_renderer.add_mesh(mesh);

        let camera = engine.scene_renderer.camera_mut();
        camera.set_perspective(70.0, 4.0 / 3.0, 0.01, 100.0);
        camera.set_position(Vec3::new(1.0, 0.0, -3.0));

        let num_x = 10;
        let num_z = 100;
        for i in 0..num_x {
            let x = i as f32 * 2.7;

            for j in 0..num_z {
                let z = j as f32 * 2.7;

                engine.scene.create_entity("TestEntity1")
                    .add_component(render_component_1.clone())
                    .add_component(*Transform::new()
                        .translate(Vec3::new(x, 0.0, z))
                        .rotate_z(f32::to_radians(30.0)));
            }
        }

        let num_x = 10;
        let num_z = 100;
        for i in 0..num_x {
            let x = i as f32;

            for j in 0..num_z {
                let z = j as f32;
                
                self.add_test_entity(&mut engine.scene, Vec3::new(x, 2.0, z));
            }
        }

        Ok(())
    }

    fn pre_render(
        &mut self,
        ticker: &mut Ticker,
        engine: &mut Engine,
        _cmd_buf: &mut PrimaryCommandBuffer,
    ) -> Result<()> {
        if engine
            .graphics
            .event_bus()
            .has_any_opt(&mut self.event_recreate_swapchain)
        {
            self.on_recreate_swapchain(engine)?;
        }

        if let Some(event) = engine
            .window
            .event_bus()
            .read_one_opt(&mut self.event_window_resized)
        {
            let aspect_ratio = event.width as f32 / event.height as f32;
            let camera = engine.scene_renderer.camera_mut();
            camera.set_aspect_ratio(aspect_ratio);
        }

        if ticker.time_since_last_dbg() >= ticker.debug_interval() {

            let stats = ticker.calculate_profiling_statistics();

            debug!("{stats:?}");

            self.debug_stats.push(stats);



            if let Some(stats) = engine.graphics.debug_pipeline_statistics() {
                debug!("{stats:?}");
            }
        }

        Ok(())
    }

    fn render(
        &mut self,
        ticker: &mut Ticker,
        engine: &mut Engine,
        _cmd_buf: &mut PrimaryCommandBuffer,
    ) -> Result<()> {
        if engine.graphics.state().first_frame() {
            debug!("FIRST FRAME!")
        }

        let window = &mut engine.window;

        if window.input().key_pressed(Key::F1) {
            engine.scene_renderer.set_wireframe_mode(
                match engine.scene_renderer.wireframe_mode() {
                    WireframeMode::Solid => WireframeMode::Both,
                    WireframeMode::Both => WireframeMode::Wire,
                    WireframeMode::Wire => WireframeMode::Solid,
                },
            );

            debug!(
                "Changed render mode: {:?}",
                engine.scene_renderer.wireframe_mode()
            );
        }

        if window.input().key_pressed(Key::Escape) {
            window.set_mouse_grabbed(!window.is_mouse_grabbed())
        }

        if window.is_mouse_grabbed() {

            let camera = engine.scene_renderer.camera_mut();

            let mouse_motion = window.input().relative_mouse_pos();
            let delta_pitch = mouse_motion.y * 0.04;
            let delta_yaw = mouse_motion.x * 0.04;
            self.camera_pitch = f32::clamp(self.camera_pitch + delta_pitch, -90.0, 90.0);
            self.camera_yaw += delta_yaw;
            if self.camera_yaw > 180.0 {
                self.camera_yaw -= 360.0;
            }
            if self.camera_yaw < -180.0 {
                self.camera_yaw += 360.0;
            }

            let scroll = window.input().mouse_scroll_amount().y;

            if scroll > 0.0 {
                self.move_speed *= 1.5;
                info!("Scroll up - move speed: {}", self.move_speed);
            } else if scroll < 0.0 {
                self.move_speed /= 1.5;
                info!("Scroll down - move speed: {}", self.move_speed);
            }

            if window.input().mouse_pressed(MouseButton::Left) {
                self.add_test_entity(&mut engine.scene, camera.position() + camera.z_axis() * 2.0);
            }

            let up_axis = Vec3::Y;
            let right_axis = camera.x_axis();
            let forward_axis = Vec3::cross(camera.x_axis(), up_axis);

            camera.set_rotation_euler(self.camera_pitch, self.camera_yaw, 0.0);

            let mut move_dir = Vec3::ZERO;
            if window.input().key_down(Key::W) {
                move_dir += forward_axis;
            }
            if window.input().key_down(Key::S) {
                move_dir -= forward_axis;
            }
            if window.input().key_down(Key::A) {
                move_dir -= right_axis;
            }
            if window.input().key_down(Key::D) {
                move_dir += right_axis;
            }
            if window.input().key_down(Key::Space) {
                move_dir += up_axis;
            }
            if window.input().key_down(Key::LShift) {
                move_dir -= up_axis;
            }

            if move_dir.length_squared() > 0.001 {
                let move_speed = self.move_speed * ticker.delta_time() as f32;
                move_dir = Vec3::normalize(move_dir) * move_speed;
                camera.move_position(move_dir);
            }
        }

        Ok(())
    }

    fn shutdown(&mut self) {
        let mut file = String::new();
        for stat in &self.debug_stats {
            let s = format!("{stat:?}\n");
            file.push_str(s.as_str())
        }

        if let Err(e) = fs::write("./debug_stats.txt", file) {
            error!("Failed to write performance statistics: {e}");
        }
    }

    fn is_stopped(&self) -> bool {
        false
    }
}

fn main() {
    Engine::start(TestGame::new());
}
