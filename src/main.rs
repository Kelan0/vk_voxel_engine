mod application;
mod core;
mod util;

use crate::application::Key;
use crate::core::{GraphicsManager, GraphicsPipeline, GraphicsPipelineConfiguration, Mesh, MeshConfiguration, PrimaryCommandBuffer, RecreateSwapchainEvent};
use anyhow::{anyhow, Result};
use application::ticker::Ticker;
use application::App;
use core::Engine;
use foldhash::{HashMap};
use log::{debug, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::{DynamicState, PipelineCreateFlags};
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

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
    mesh: Option<Mesh<BaseVertex>>,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    wire_state: TestWireframeState,
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
            mesh: None,
            event_recreate_swapchain: None,
            wire_state: TestWireframeState::Solid,
        }
    }

    fn on_recreate_swapchain(&mut self, engine: &Engine) -> Result<()> {
        self.init_pipeline_test(engine)
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
}


impl App for TestGame {
    fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        Ok(())
    }

    fn init(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()> {
        info!("Init TestGame");
        let window = &mut engine.window;
        ticker.set_desired_tick_rate(60.0);
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

        Ok(())
    }

    fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }

        if ticker.time_since_last_dbg() >= 1.0 {
            let stats = ticker.calculate_profiling_statistics();

            debug!("{stats:?}");
        }
        Ok(())
    }

    fn render(&mut self, _ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        if engine.graphics.state().first_frame() {
            debug!("FIRST FRAME!")
        }


        let window = &mut engine.window;

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

        if window.input().key_down(Key::W) {
            debug!("Move");
        }


        let viewport = Viewport{
            offset: [0.0, 0.0],
            extent: engine.window.get_window_size_in_pixels().into(),
            depth_range: 0.0..=1.0,
        };

        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;


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
