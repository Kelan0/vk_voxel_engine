use std::fs::File;
use std::io::Read;
use crate::core::{AxisDirection, BaseVertex, CommandBuffer, Engine, FrameCompleteEvent, GraphicsManager, GraphicsPipelineBuilder, RecreateSwapchainEvent, RenderCamera, Texture};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use foldhash::HashMap;
use log::{debug, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineCreateFlags};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::render_pass::Subpass;
use crate::application::Ticker;
use crate::{function_name, profile_scope_fn};

#[derive(BufferContents, Vertex, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct VoxelVertex {
    #[format(R32_UINT)]
    pub vs_data: u32, // XXXXXYYYYYZZZZZNNN // X[5] Y[5] Z[5] Norm[3]
}

impl VoxelVertex {
    const MASK_X: u32 = 0b11111_00000_00000_000;
    const SHIFT_X: u32 = 5+5+3;
    const MASK_Y: u32 = 0b00000_11111_00000_000;
    const SHIFT_Y: u32 = 5+3;
    const MASK_Z: u32 = 0b00000_00000_11111_000;
    const SHIFT_Z: u32 = 3;
    const MASK_NORM: u32 = 0b00000_00000_00000_111;
    const SHIFT_NORM: u32 = 0;

    pub fn from_packed(data: u32) -> Self {
        VoxelVertex {
            vs_data: data
        }
    }

    pub fn from_unpacked(x: u32, y: u32, z: u32, dir: AxisDirection) -> Self {
        let data = Self::pack_data(x, y, z, dir);
        Self::from_packed(data)
    }

    pub fn pack_data(x: u32, y: u32, z: u32, dir: AxisDirection) -> u32 {
        (x & (Self::MASK_X >> Self::SHIFT_X) << Self::SHIFT_X) |
            (y & (Self::MASK_Y >> Self::SHIFT_Y) << Self::SHIFT_Y) |
            (z & (Self::MASK_Z >> Self::SHIFT_Z) << Self::SHIFT_Z) |
            (dir.index() * (Self::MASK_NORM >> Self::SHIFT_NORM) << Self::SHIFT_NORM)
    }

    pub fn unpack_x(data: u32) -> u32 {
        (data & Self::MASK_X) >> Self::SHIFT_X
    }

    pub fn unpack_y(data: u32) -> u32 {
        (data & Self::MASK_Y) >> Self::SHIFT_Y
    }

    pub fn unpack_z(data: u32) -> u32 {
        (data & Self::MASK_Z) >> Self::SHIFT_Z
    }

    pub fn unpack_dir(data: u32) -> u32 {
        (data & Self::MASK_NORM) >> Self::SHIFT_NORM
    }

    pub fn packed_data(&self) -> u32 {
        self.vs_data
    }
}

pub struct VoxelRenderer {
    solid_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    wire_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    resources: Vec<FrameResource>,

    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
}

struct FrameResource {
    descriptor_set_camera: Option<Arc<DescriptorSet>>,

    recreate_descriptor_sets: bool,
    camera_hash: u64,
}

impl VoxelRenderer {
    pub fn new(graphics: &GraphicsManager) -> Result<Self> {
        let voxel_renderer = VoxelRenderer {
            solid_graphics_pipeline: None,
            wire_graphics_pipeline: None,
            resources: vec![],

            event_recreate_swapchain: None,
        };

        Ok(voxel_renderer)
    }


    pub fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        Ok(())
    }

    pub fn init(&mut self, _engine: &mut Engine) -> Result<()> {
        Ok(())
    }

    fn init_resources(&mut self, engine: &mut Engine) -> Result<()> {

        self.resources.resize_with(engine.graphics.max_concurrent_frames(), || {
            FrameResource{
                descriptor_set_camera: None,

                recreate_descriptor_sets: true,
                camera_hash: 0,
            }
        });

        Ok(())
    }
    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        debug!("VoxelRenderer - Recreate swapchain");
        self.create_main_graphics_pipeline(engine)?;

        self.init_resources(engine)?;

        for resource in self.resources.iter_mut() {
            resource.recreate_descriptor_sets = true;
        }

        Ok(())
    }
    fn create_main_graphics_pipeline(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize VoxelRenderer GraphicsPipeline");
        let device = engine.graphics.device();
        let render_pass = engine.graphics.render_pass();

        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/world_solid.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "world_solid.glsl::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("FRAGMENT_SHADER_MODULE", None);
        let fs_solid = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "world_solid.glsl::frag(solid)", "main", ShaderKind::Fragment, Some(&options))?;

        options.add_macro_definition("WIREFRAME_ENABLED", None);
        let fs_wire = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "world_solid.glsl::frag(wire)", "main", ShaderKind::Fragment, Some(&options))?;

        let subpass_type = Subpass::from(render_pass.clone(), 0)
            .ok_or_else(|| anyhow!("Failed to get subpass info for provided RenderPass"))?
            .into();

        let entry_points = HashMap::from_iter([(0, "main"), (1, "main") ]);

        let main_graphics_pipeline = GraphicsPipelineBuilder::new(subpass_type, vec![vs.clone(), fs_solid], entry_points.clone())
            .add_flags(PipelineCreateFlags::ALLOW_DERIVATIVES)
            .set_dynamic_states(vec![ DynamicState::Viewport ])
            .set_input_assembly_state(InputAssemblyState{
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            })
            .set_viewport_state(ViewportState::default())
            .set_multisample_state(MultisampleState::default())
            .set_rasterization_state(RasterizationState{
                polygon_mode: PolygonMode::Fill,
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            })
            .set_depth_stencil_state(DepthStencilState{
                depth: Some(DepthState{ write_enable: true, compare_op: CompareOp::Less }),
                // depth_bounds: Some(0.0..=1.0),
                ..Default::default()
            })
            .set_color_blend_state(ColorBlendState{
                attachments: vec![ColorBlendAttachmentState::default()],
                ..Default::default()
            })
            .build_pipeline::<BaseVertex>(device.clone())?;

        let wire_graphics_pipeline = GraphicsPipelineBuilder::new_derive(main_graphics_pipeline.clone(), vec![vs.clone(), fs_wire], entry_points.clone())
            .set_rasterization_state(RasterizationState{
                polygon_mode: PolygonMode::Line,
                cull_mode: CullMode::None,
                line_width: 1.0,
                depth_bias: Some(DepthBiasState{ constant_factor: 1.0, clamp: 0.0, slope_factor: -1.0 }),
                ..main_graphics_pipeline.rasterization_state().clone()
            })
            .set_depth_stencil_state(DepthStencilState{
                depth: Some(DepthState{ write_enable: true, compare_op: CompareOp::Less }),
                // depth_bounds: Some(0.0..=1.0),
                ..Default::default()
            })
            .build_pipeline::<BaseVertex>(device.clone())?;


        self.solid_graphics_pipeline = Some(main_graphics_pipeline);
        self.wire_graphics_pipeline = Some(wire_graphics_pipeline);

        Ok(())
    }

    pub fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut CommandBuffer) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);

        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }
        Ok(())
    }

    pub fn render(&mut self, camera: &RenderCamera, _ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if Self::check_changed_camera(camera, resource, frame_index) {
            Self::create_camera_descriptor_sets(&camera, resource, self.solid_graphics_pipeline.as_ref().unwrap(), &engine.graphics)?;
            resource.camera_hash = camera.gpu_resource_hash(frame_index);
        }
        
        Ok(())
    }

    fn check_changed_camera(camera: &RenderCamera, resource: &FrameResource, frame_index: usize) -> bool {
        camera.gpu_resource_hash(frame_index) != resource.camera_hash
    }

    fn create_camera_descriptor_sets(camera: &RenderCamera, resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {

        let descriptor_set = camera.create_camera_descriptor_sets(0, 0, graphics_pipeline, graphics)?;
        resource.descriptor_set_camera = Some(descriptor_set);

        Ok(())
    }
}
