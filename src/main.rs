mod application;
mod core;
mod util;

use crate::application::Key;
use crate::core::{PrimaryCommandBuffer, RecreateSwapchainEvent};
use anyhow::{anyhow, Result};
use application::ticker::Ticker;
use application::App;
use core::Engine;
use foldhash::HashSet;
use log::{debug, info};
use shaderc::{CompilationArtifact, ShaderKind};
use smallvec::smallvec;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::time::{Duration, Instant};
use shrev::ReaderId;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::tessellation::TessellationState;
use vulkano::pipeline::graphics::vertex_input::{BuffersDefinition, Vertex, VertexDefinition, VertexInputBindingDescription, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineCreateFlags, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo};

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct BaseVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

struct TestGame {
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer: Option<Subbuffer<[BaseVertex]>>,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>
}

impl TestGame {
    fn new() -> Self {
        TestGame {
            graphics_pipeline: None,
            vertex_buffer: None,
            event_recreate_swapchain: None,
        }
    }

    fn on_recreate_swapchain(&mut self, engine: &Engine) {
        self.init_pipeline_test(engine).expect("Failed to create GraphicsPipeline");
    }

    fn init_pipeline_test(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize GraphicsPipeline");
        let device = engine.graphics.get_device();
        let render_pass = engine.graphics.get_render_pass();


        let vs = {
            let compiled_code = compile_to_spirv("res/shaders/test_vert.glsl", ShaderKind::Vertex, "main");
            let bytes = compiled_code.as_binary_u8();
            let shader_code = vulkano::shader::spirv::bytes_to_words(&bytes)?;

            let shader_create_info = ShaderModuleCreateInfo::new(&shader_code);
            let shader_module = unsafe { ShaderModule::new(device.clone(), shader_create_info) }?;
            shader_module.entry_point("main").unwrap()
        };
        let fs = {
            let compiled_code = compile_to_spirv("res/shaders/test_frag.glsl", ShaderKind::Fragment, "main");
            let bytes = compiled_code.as_binary_u8();
            let shader_code = vulkano::shader::spirv::bytes_to_words(&bytes)?;

            let shader_create_info = ShaderModuleCreateInfo::new(&shader_code);
            let shader_module = unsafe { ShaderModule::new(device.clone(), shader_create_info) }?;
            shader_module.entry_point("main").unwrap()
        };

        let stages = smallvec![
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];

        let create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages).into_pipeline_layout_create_info(device.clone())?;

        let vertex_input_state = BaseVertex::per_vertex().definition(&vs)?;

        let input_assembly_state = InputAssemblyState{
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        };

        let tessellation_state = TessellationState{
            ..Default::default()
        };

        let viewport_state = ViewportState{
            ..Default::default()
        };

        let rasterization_state = RasterizationState{
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::None,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let multisample_state = MultisampleState{
            ..Default::default()
        };

        let color_blend_state = ColorBlendState{
            attachments: vec![ColorBlendAttachmentState::default()],
            ..Default::default()
        };

        let dynamic_state = HashSet::from_iter([
            DynamicState::Viewport
        ]);

        let subpass = Subpass::from(render_pass.clone(), 0)
            .ok_or_else(|| anyhow!("Failed to get subpass info for provided RenderPass"))?
            .into();

        let pipeline_layout = PipelineLayout::new(device.clone(), create_info)?;

        let create_info = GraphicsPipelineCreateInfo{
            stages,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(input_assembly_state),
            // tessellation_state: Some(tessellation_state),
            viewport_state: Some(viewport_state),
            rasterization_state: Some(rasterization_state),
            multisample_state: Some(multisample_state),
            depth_stencil_state: None,
            color_blend_state: Some(color_blend_state),
            dynamic_state,
            subpass: Some(subpass),
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        let pipeline = GraphicsPipeline::new(device, None, create_info)?;

        self.graphics_pipeline = Some(pipeline);

        Ok(())
    }
}

fn compile_to_spirv(src: &str, kind: ShaderKind, entry_point_name: &str) -> CompilationArtifact {
    let mut f = File::open(src).unwrap_or_else(|_| panic!("Could not open file {}", src));
    let mut glsl = String::new();
    f.read_to_string(&mut glsl)
        .unwrap_or_else(|_| panic!("Could not read file {} to string", src));

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.add_macro_definition("EP", Some(entry_point_name));
    compiler
        .compile_into_spirv(&glsl, kind, src, entry_point_name, Some(&options))
        .expect("Could not compile glsl shader to spriv")
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
            BaseVertex {
                position: [-0.5, 0.5],
            },
            BaseVertex {
                position: [0.5, 0.5],
            },
            BaseVertex {
                position: [0.0, -0.5],
            },
        ];

        let allocator = engine.graphics.get_memory_allocator();
        let buffer_create_info = BufferCreateInfo{
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let vertex_buffer = Buffer::from_iter(allocator, buffer_create_info, allocation_info, vertices)?;
        self.vertex_buffer = Some(vertex_buffer);

        Ok(())
    }

    fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine);
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

        let graphics_pipeline = self.graphics_pipeline.as_ref().unwrap().clone();
        let vertex_buffer = self.vertex_buffer.as_ref().unwrap().clone();
        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;
        cmd_buf.bind_pipeline_graphics(graphics_pipeline)?;
        cmd_buf.bind_vertex_buffers(0, vertex_buffer.clone())?;
        unsafe { cmd_buf.draw(vertex_buffer.len() as u32, 1, 0, 0) }?;

        // if window.input().mouse_dragged(MouseBtn::Left) {
        //     debug!("Mouse dragged {}", window.input().get_mouse_drag_pixel_distance(MouseBtn::Left));
        // } else {
        //     let vec = window.input().get_mouse_pixel_motion();
        //     if vec.x != 0.0 && vec.y != 0.0 {
        //         debug!("Mouse motion: {vec}");
        //     }
        // }

        Ok(())
    }

    fn is_stopped(&self) -> bool {
        false
    }
}

fn main() {
    Engine::start(TestGame::new());
}
