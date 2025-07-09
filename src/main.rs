mod application;
mod core;
mod util;

use crate::application::Key;
use crate::core::{Camera, CameraBufferUBO, GraphicsManager, GraphicsPipeline, GraphicsPipelineConfiguration, Mesh, MeshConfiguration, PrimaryCommandBuffer, RecreateSwapchainEvent};
use anyhow::{anyhow, Result};
use application::ticker::Ticker;
use application::App;
use core::Engine;
use foldhash::{HashMap};
use log::{debug, error, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use glam::{Mat4, Quat, Vec3, Vec4};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, BufferWriteGuard, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags};
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::HostAccessError;
use crate::application::window::WindowResizedEvent;

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct BaseVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    colour: [f32; 3],
}


struct TestGame {
    solid_graphics_pipeline: Option<GraphicsPipeline>,
    wire_graphics_pipeline: Option<GraphicsPipeline>,
    uniform_buffer_camera: Option<Subbuffer<CameraBufferUBO>>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    mesh: Option<Mesh<BaseVertex>>,
    camera: Camera,
    camera_pitch: f32,
    camera_yaw: f32,
    wire_state: TestWireframeState,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_window_resized: Option<ReaderId<WindowResizedEvent>>,
}

#[derive(Debug, Eq, PartialEq)]
enum TestWireframeState {
    Solid,
    Wire,
    Both,
}

impl TestGame {
    fn new() -> Self {
        TestGame {
            solid_graphics_pipeline: None,
            wire_graphics_pipeline: None,
            uniform_buffer_camera: None,
            descriptor_set: None,
            mesh: None,
            camera: Camera::default(),
            camera_pitch: 0.0,
            camera_yaw: 0.0,
            wire_state: TestWireframeState::Solid,
            event_recreate_swapchain: None,
            event_window_resized: None,
        }
    }

    fn on_recreate_swapchain(&mut self, engine: &Engine) -> Result<()> {
        self.init_pipeline_test(engine)?;
        self.init_descriptor_sets(engine)?;
        Ok(())
    }

    fn init_pipeline_test(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize GraphicsPipeline");
        let device = engine.graphics.get_device();
        let render_pass = engine.graphics.get_render_pass();


        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/test.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test_solid::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("FRAGMENT_SHADER_MODULE", None);
        let fs_solid = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test_solid::frag(solid)", "main", ShaderKind::Fragment, Some(&options))?;
        
        options.add_macro_definition("WIREFRAME_ENABLED", None);
        let fs_wire = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test_solid::frag(wire)", "main", ShaderKind::Fragment, Some(&options))?;

        // let vs = engine.graphics.load_shader_module_from_file("res/shaders/test_vert.glsl", "main", ShaderKind::Vertex, None)?;
        // let fs = engine.graphics.load_shader_module_from_file("res/shaders/test_frag.glsl", "main", ShaderKind::Fragment, None)?;

        let subpass_type = Subpass::from(render_pass.clone(), 0)
            .ok_or_else(|| anyhow!("Failed to get subpass info for provided RenderPass"))?
            .into();

        let entry_points = HashMap::from_iter([(0, "main"), (1, "main") ]);

        let pipeline_config = GraphicsPipelineConfiguration{
            flags: PipelineCreateFlags::ALLOW_DERIVATIVES,
            dynamic_states: vec![ DynamicState::Viewport ],
            rasterization_state: RasterizationState{
                polygon_mode: PolygonMode::Fill,
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            },
            ..GraphicsPipelineConfiguration::new(subpass_type, vec![vs.clone(), fs_solid], entry_points, 0)
        };

        
        let main_graphics_pipeline = GraphicsPipeline::new::<BaseVertex>(device.clone(), pipeline_config.clone())?;
        
        let wire_graphics_pipeline = GraphicsPipeline::new::<BaseVertex>(device.clone(), GraphicsPipelineConfiguration{
            shader_modules: vec![vs.clone(), fs_wire],
            rasterization_state: RasterizationState{
                polygon_mode: PolygonMode::Line,
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            },
            ..pipeline_config.clone()
        })?;

        self.solid_graphics_pipeline = Some(main_graphics_pipeline);
        self.wire_graphics_pipeline = Some(wire_graphics_pipeline);

        Ok(())
    }

    fn init_descriptor_sets(&mut self, engine: &Engine) -> Result<()> {

        let memory_allocator = engine.graphics.get_memory_allocator();
        let descriptor_set_allocator = engine.graphics.get_descriptor_set_allocator();

        let buffer_create_info = BufferCreateInfo{
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let uniform_buffer_camera = Buffer::new_sized::<CameraBufferUBO>(memory_allocator.clone(), buffer_create_info, allocation_info)?;
        // let uniform_buffer_camera = Buffer::from_data(memory_allocator.clone(), buffer_create_info, allocation_info, camera_data)?;

        self.uniform_buffer_camera = Some(uniform_buffer_camera.clone());

        let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap().get();


        let pipeline_layout = solid_graphics_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_writes = [WriteDescriptorSet::buffer(0, uniform_buffer_camera)];
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), descriptor_writes, [])?;

        self.descriptor_set = Some(descriptor_set);
        Ok(())
    }
}


impl App for TestGame {
    fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        self.event_window_resized = Some(engine.window.event_bus().register::<WindowResizedEvent>());
        Ok(())
    }

    fn init(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        info!("Init TestGame");
        let window = &mut engine.window;
        ticker.set_desired_tick_rate(175.0);
        window.set_visible(true);


        let vertices = [
            BaseVertex { position: [-0.5, 0.5], colour: [0.0, 1.0, 0.0] },
            BaseVertex { position: [0.5, 0.5], colour: [1.0, 1.0, 0.0] },
            BaseVertex { position: [-0.5, -0.5], colour: [0.0, 0.0, 0.0] },
            BaseVertex { position: [0.5, -0.5], colour: [1.0, 0.0, 0.0] },
        ];

        let indices = [
            0, 1, 2,
            1, 3, 2
        ];

        let allocator = engine.graphics.get_memory_allocator();

        let mesh_config = MeshConfiguration{
            vertices: Vec::from(vertices),
            indices: Some(Vec::from(indices)),
        };

        let mesh = Mesh::new(allocator.clone(), mesh_config)?;
        self.mesh = Some(mesh);

        self.camera.set_perspective(70.0, 4.0 / 3.0, 0.01, 100.0);
        self.camera.set_position(Vec3::new(1.0, 0.0, -3.0));

        Ok(())
    }

    fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }

        if let Some(event) = engine.window.event_bus().read_one_opt(&mut self.event_window_resized) {
            let aspect_ratio = event.width as f32 / event.height as f32;
            self.camera.set_aspect_ratio(aspect_ratio);
        }

        if ticker.time_since_last_dbg() >= 1.0 {
            let stats = ticker.calculate_profiling_statistics();

            debug!("{stats:?}");
        }
        Ok(())
    }

    fn render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        if engine.graphics.state().first_frame() {
            debug!("FIRST FRAME!")
        }


        let window = &mut engine.window;
        let uniform_buffer_camera = self.uniform_buffer_camera.as_ref().unwrap();

        if window.input().key_pressed(Key::F1) {
            self.wire_state = match self.wire_state {
                TestWireframeState::Solid => TestWireframeState::Wire,
                TestWireframeState::Wire => TestWireframeState::Both,
                TestWireframeState::Both => TestWireframeState::Solid
            };

            debug!("Changed render mode: {:?}", self.wire_state);
        }

        if window.input().key_pressed(Key::Escape) {
            window.set_mouse_grabbed(!window.is_mouse_grabbed())
        }

        if window.is_mouse_grabbed() {
            let mouse_motion = window.input().relative_mouse_pos();
            let delta_pitch = mouse_motion.y * ticker.get_delta_time() as f32 * 10.0 * -1.0;
            let delta_yaw = mouse_motion.x * ticker.get_delta_time() as f32 * 10.0;
            self.camera_pitch = f32::clamp(self.camera_pitch + delta_pitch, -90.0, 90.0);
            self.camera_yaw += delta_yaw;
            if self.camera_yaw > 180.0 {
                self.camera_yaw -= 360.0;
            }
            if self.camera_yaw < -180.0 {
                self.camera_yaw += 360.0;
            }

            let right_axis = self.camera.x_axis();
            let forward_axis = Vec3::cross(self.camera.x_axis(), Vec3::Y);

            self.camera.set_rotation_euler(self.camera_pitch, self.camera_yaw, 0.0);

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

            if move_dir.length_squared() > 0.001 {
                let move_speed = 1.5 * ticker.get_delta_time() as f32;
                move_dir = Vec3::normalize(move_dir) * move_speed;
                self.camera.move_position(move_dir);
            }
        }


        self.camera.update();

        match uniform_buffer_camera.write() {
            Ok(mut write) => self.camera.update_camera_buffer(&mut *write),
            Err(err) => error!("Unable to write camera data: {}", err)
        }

        let viewport = Viewport{
            offset: [0.0, 0.0],
            extent: engine.window.get_window_size_in_pixels().into(),
            depth_range: 0.0..=1.0,
        };

        let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap().get();
        let descriptor_set = self.descriptor_set.as_ref().unwrap();
        let pipeline_layout = solid_graphics_pipeline.layout();

        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;
        cmd_buf.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![descriptor_set.clone()])?;

        if self.wire_state == TestWireframeState::Solid || self.wire_state == TestWireframeState::Both {
            let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(solid_graphics_pipeline.get())?;
            self.mesh.as_ref().unwrap().draw(cmd_buf, 1, 0)?;
        }
        if self.wire_state == TestWireframeState::Wire || self.wire_state == TestWireframeState::Both {
            let wire_graphics_pipeline = self.wire_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(wire_graphics_pipeline.get())?;
            self.mesh.as_ref().unwrap().draw(cmd_buf, 1, 0)?;
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
