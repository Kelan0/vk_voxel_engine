mod application;
mod core;
mod util;

use crate::application::window::WindowResizedEvent;
use crate::application::Key;
use crate::core::{BaseVertex, Mesh, MeshConfiguration, PrimaryCommandBuffer, RecreateSwapchainEvent, RenderComponent, Transform, WireframeMode};
use anyhow::Result;
use application::ticker::Ticker;
use application::App;
use core::Engine;
use glam::{Affine3A, DAffine3, Vec3};
use log::{debug, info};
use shrev::ReaderId;

struct TestGame {
    camera_pitch: f32,
    camera_yaw: f32,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_window_resized: Option<ReaderId<WindowResizedEvent>>,
}

impl TestGame {
    fn new() -> Self {
        TestGame {
            camera_pitch: 0.0,
            camera_yaw: 0.0,
            event_recreate_swapchain: None,
            event_window_resized: None,
        }
    }

    fn on_recreate_swapchain(&mut self, _engine: &Engine) -> Result<()> {
        Ok(())
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
        ticker.set_desired_tick_rate(175.0);
        window.set_visible(true);

        let vertices = [
            BaseVertex {
                position: [-0.5, 1.5],
                colour: [0.0, 1.0, 0.0],
            },
            BaseVertex {
                position: [0.5, 0.5],
                colour: [1.0, 1.0, 0.0],
            },
            BaseVertex {
                position: [-0.5, -0.5],
                colour: [0.0, 0.0, 0.0],
            },
            BaseVertex {
                position: [0.5, -0.5],
                colour: [1.0, 0.0, 0.0],
            },
        ];

        let indices = [0, 1, 2, 1, 3, 2];

        let allocator = engine.graphics.memory_allocator();

        let mesh_config = MeshConfiguration {
            vertices: Vec::from(vertices),
            indices: Some(Vec::from(indices)),
        };

        let mesh = Mesh::new(allocator.clone(), mesh_config)?;
        let render_component = RenderComponent::new(mesh);

        // engine.scene_renderer.add_mesh(mesh);

        let camera = engine.scene_renderer.camera_mut();
        camera.set_perspective(70.0, 4.0 / 3.0, 0.01, 100.0);
        camera.set_position(Vec3::new(1.0, 0.0, -3.0));

        for i in 0..100 {
            let x = i as f32;
            
            for j in 0..100 {
                let z = j as f32;

                engine.scene.create_entity("TestEntity")
                    .add_component(render_component.clone())
                    .add_component(Transform::new()
                        .translate(Vec3::new(x, 0.0, z))
                        .rotate_z(f32::to_radians(30.0))
                        .clone());
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

        if ticker.time_since_last_dbg() >= 1.0 {
            let stats = ticker.calculate_profiling_statistics();

            debug!("{stats:?}");
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
                    WireframeMode::Solid => WireframeMode::Wire,
                    WireframeMode::Wire => WireframeMode::Both,
                    WireframeMode::Both => WireframeMode::Solid,
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
            let mouse_motion = window.input().relative_mouse_pos();
            let delta_pitch = mouse_motion.y * ticker.delta_time() as f32 * 10.0;
            let delta_yaw = mouse_motion.x * ticker.delta_time() as f32 * 10.0;
            self.camera_pitch = f32::clamp(self.camera_pitch + delta_pitch, -90.0, 90.0);
            self.camera_yaw += delta_yaw;
            if self.camera_yaw > 180.0 {
                self.camera_yaw -= 360.0;
            }
            if self.camera_yaw < -180.0 {
                self.camera_yaw += 360.0;
            }

            let camera = engine.scene_renderer.camera_mut();

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
                let move_speed = 1.5 * ticker.delta_time() as f32;
                move_dir = Vec3::normalize(move_dir) * move_speed;
                camera.move_position(move_dir);
            }
        }

        Ok(())
    }

    fn is_stopped(&self) -> bool {
        false
    }
}

fn main() {
    Engine::start(TestGame::new());
}
