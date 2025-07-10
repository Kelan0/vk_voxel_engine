use std::fs::File;
use std::io::Read;
use std::mem;
use std::sync::Arc;
use anyhow::anyhow;
use anyhow::Result;
use foldhash::HashMap;
use log::{error, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineCreateFlags};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::render_pass::Subpass;
use crate::application::Ticker;
use crate::core::{Camera, CameraBufferUBO, Engine, GraphicsPipelineBuilder, Mesh, PrimaryCommandBuffer, RecreateSwapchainEvent};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct BaseVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    pub colour: [f32; 3],
}

pub struct SceneRenderer {
    solid_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    wire_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    resources: Vec<FrameResource>,
    wireframe_mode: WireframeMode,
    camera: Camera,

    meshes: Vec<Mesh<BaseVertex>>, // temporary

    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
}

struct FrameResource {
    uniform_buffer_camera: Option<Subbuffer<CameraBufferUBO>>,
    descriptor_set: Option<Arc<DescriptorSet>>,
    update_descriptor_sets: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WireframeMode {
    Solid,
    Wire,
    Both,
}


impl SceneRenderer {
    pub fn new() -> Result<Self> {

        let scene_renderer = SceneRenderer{
            solid_graphics_pipeline: None,
            wire_graphics_pipeline: None,
            resources: vec![],
            wireframe_mode: WireframeMode::Solid,
            camera: Camera::new(),

            meshes: vec![],

            event_recreate_swapchain: None,
        };

        Ok(scene_renderer)
    }

    pub fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        Ok(())
    }


    pub fn init(&mut self, engine: &mut Engine) -> Result<()> {
        self.resources.resize_with(engine.graphics.max_concurrent_frames(), || FrameResource{
            uniform_buffer_camera: None,
            descriptor_set: None,
            update_descriptor_sets: false,
        });
        Ok(())
    }

    pub fn pre_render(&mut self, _ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }

        self.camera.update();
        
        // info!("Camera position: {:?}, Direction: {:?} - fov={}, aspect={}, far={}", self.camera.position(), self.camera.z_axis(), self.camera.fov(), self.camera.aspect_ratio(), self.camera.far_plane());



        Ok(())
    }

    pub fn render(&mut self, _ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        let viewport = engine.graphics.get_viewport();

        let resource = Self::curr_resource(&mut self.resources, engine);

        let uniform_buffer_camera = resource.uniform_buffer_camera.as_ref().unwrap();

        match uniform_buffer_camera.write() {
            Ok(mut write) => self.camera.update_camera_buffer(&mut write),
            Err(err) => error!("Unable to write camera data: {err}")
        }
        
        
        let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
        let descriptor_set = resource.descriptor_set.as_ref().unwrap();
        let pipeline_layout = solid_graphics_pipeline.layout();

        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;
        cmd_buf.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![descriptor_set.clone()])?;

        if self.wireframe_mode == WireframeMode::Solid || self.wireframe_mode == WireframeMode::Both {
            let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(solid_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf)?;
        }
        if self.wireframe_mode == WireframeMode::Wire || self.wireframe_mode == WireframeMode::Both {
            let wire_graphics_pipeline = self.wire_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(wire_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf)?;
        }

        Ok(())
    }

    fn draw_scene(&self, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        for mesh in self.meshes.iter() {
            mesh.draw(cmd_buf, 1, 0)?;
        }
        Ok(())
    }

    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        self.create_main_graphics_pipeline(engine)?;

        let mut resources = mem::take(&mut self.resources);
        for resource in resources.iter_mut() {
            self.create_descriptor_sets(engine, resource)?
        }
        self.resources = resources;

        Ok(())
    }

    fn create_main_graphics_pipeline(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize GraphicsPipeline");
        let device = engine.graphics.device();
        let render_pass = engine.graphics.render_pass();

        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/test.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test.glsl::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("FRAGMENT_SHADER_MODULE", None);
        let fs_solid = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test.glsl::frag(solid)", "main", ShaderKind::Fragment, Some(&options))?;

        options.add_macro_definition("WIREFRAME_ENABLED", None);
        let fs_wire = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "test.glsl::frag(wire)", "main", ShaderKind::Fragment, Some(&options))?;

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
                cull_mode: CullMode::None,
                front_face: FrontFace::CounterClockwise,
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
                line_width: 2.0,
                ..main_graphics_pipeline.rasterization_state().clone()
            })
            .build_pipeline::<BaseVertex>(device.clone())?;

        self.solid_graphics_pipeline = Some(main_graphics_pipeline);
        self.wire_graphics_pipeline = Some(wire_graphics_pipeline);

        Ok(())
    }

    fn create_descriptor_sets(&self, engine: &Engine, resource: &mut FrameResource) -> Result<()> {

        if self.solid_graphics_pipeline.is_none() {
            return Err(anyhow!("SceneRenderer - Unable to create descriptor sets because graphics pipeline is None"));
        }

        let memory_allocator = engine.graphics.memory_allocator();
        let descriptor_set_allocator = engine.graphics.descriptor_set_allocator();

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


        let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();


        let pipeline_layout = solid_graphics_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_writes = [WriteDescriptorSet::buffer(0, uniform_buffer_camera.clone())];
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), descriptor_writes, [])?;


        resource.uniform_buffer_camera = Some(uniform_buffer_camera.clone());
        resource.descriptor_set = Some(descriptor_set);
        resource.update_descriptor_sets = true;

        Ok(())
    }

    fn curr_resource<'a>(resources: &'a mut [FrameResource], engine: &Engine) -> &'a mut FrameResource {
        Self::resource(resources, engine.graphics.current_frame_index())
    }

    fn resource(resources: &mut [FrameResource], index: usize) -> &mut FrameResource {
        // self.resources[index].as_mut().unwrap()
        &mut resources[index]
    }

    pub fn add_mesh(&mut self, mesh: Mesh<BaseVertex>) {
        self.meshes.push(mesh);
    }

    pub fn set_wireframe_mode(&mut self, wireframe_mode: WireframeMode) {
        self.wireframe_mode = wireframe_mode;
    }

    pub fn wireframe_mode(&self) -> WireframeMode {
        self.wireframe_mode
    }
    
    pub fn camera(&self) -> &Camera {
        &self.camera
    }
    
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
}