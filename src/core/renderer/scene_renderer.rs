use crate::application::Ticker;
use crate::core::{Camera, CameraDataUBO, Engine, GraphicsManager, GraphicsPipelineBuilder, Mesh, PrimaryCommandBuffer, RecreateSwapchainEvent, RenderComponent, RenderType, Scene, Transform, VertexHasColour, VertexHasNormal, VertexHasPosition};
use anyhow::anyhow;
use anyhow::Result;
use foldhash::HashMap;
use log::{debug, error, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::sync::Arc;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Added;
use bevy_ecs::query::With;
use glam::Vec3;
use rayon::slice::ParallelSliceMut;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineCreateFlags};
use vulkano::render_pass::Subpass;
use vulkano::DeviceSize;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};

#[derive(BufferContents, Vertex, Clone, PartialEq)]
#[repr(C)]
pub struct BaseVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub colour: [f32; 3],
}

impl BaseVertex {
    pub fn new(position: Vec3, normal: Vec3, colour: Vec3) -> Self {
        BaseVertex {
            position: [ position.x, position.y, position.z ],
            normal: [ normal.x, normal.y, normal.z ],
            colour: [ colour.x, colour.y, colour.z ]
        }
    }
}

impl Default for BaseVertex {
    fn default() -> Self {
        BaseVertex {
            position: [0.0; 3],
            normal: [0.0; 3],
            colour: [1.0; 3],
        }
    }
}

impl VertexHasPosition<f32> for BaseVertex {
    fn position(&self) -> &[f32; 3] {
        &self.position
    }

    fn set_position(&mut self, pos: [f32; 3]) {
        self.position = pos;
    }
}

impl VertexHasNormal<f32> for BaseVertex {
    fn normal(&self) -> &[f32; 3] {
        &self.normal
    }

    fn set_normal(&mut self, normal: [f32; 3]) {
        self.normal = normal
    }
}

impl VertexHasColour<f32> for BaseVertex {
    fn colour(&self) -> &[f32; 3] {
        &self.colour
    }

    fn set_colour(&mut self, colour: [f32; 3]) {
        self.colour = colour;
    }
}


#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct ObjectDataUBO {
    model_matrix: [f32; 16]
    //    uint materialIndex;
    //    uint _pad0;
    //    uint _pad1;
    //    uint _pad2;
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct ObjectIndexUBO {
    index: u32
}

struct RenderInfo {
    mesh: Arc<Mesh<BaseVertex>>,
    index: u32,
}

#[derive(Clone)]
struct BatchedDrawCommand {
    mesh: Arc<Mesh<BaseVertex>>,
    first_instance: u32,
    instance_count: u32,
}

#[derive(Component)]
struct StaticRenderComponentMarker;
#[derive(Component)]
struct DynamicRenderComponentMarker;


pub struct SceneRenderer {
    graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    wire_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    resources: Vec<FrameResource>,
    wireframe_mode: WireframeMode,
    camera: Camera,
    // meshes: Vec<Mesh<BaseVertex>>, // temporary

    render_info: Vec<RenderInfo>,
    object_data: Vec<ObjectDataUBO>,
    object_indices: Vec<ObjectIndexUBO>,
    static_object_count: u32,
    dynamic_object_count: u32,
    max_object_count: u32,
    static_scene_changed: bool,

    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
}

struct FrameResource {
    buffer_camera_uniforms: Option<Subbuffer<CameraDataUBO>>,
    buffer_object_data: Option<Subbuffer<[ObjectDataUBO]>>,
    buffer_object_indices: Option<Subbuffer<[ObjectIndexUBO]>>,
    descriptor_set_camera: Option<Arc<DescriptorSet>>,
    descriptor_set_world: Option<Arc<DescriptorSet>>,
    descriptor_writes_world: Vec<WriteDescriptorSet>,
    recreate_descriptor_sets: bool,
    static_scene_changed: bool,
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

            graphics_pipeline: None,
            wire_graphics_pipeline: None,
            resources: vec![],
            wireframe_mode: WireframeMode::Solid,
            camera: Camera::new(),

            render_info: vec![],
            object_data: vec![],
            object_indices: vec![],
            // meshes: vec![],

            static_object_count: 0,
            dynamic_object_count: 0,
            max_object_count: 100,
            static_scene_changed: false,

            event_recreate_swapchain: None,
        };

        Ok(scene_renderer)
    }

    pub fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        Ok(())
    }


    pub fn init(&mut self, engine: &mut Engine) -> Result<()> {
        Ok(())
    }
    
    fn init_resources(&mut self, engine: &mut Engine) -> Result<()> {

        let memory_allocator = engine.graphics.memory_allocator();

        self.resources.resize_with(engine.graphics.max_concurrent_frames(), || {
            FrameResource{
                buffer_camera_uniforms: None,
                buffer_object_data: None,
                buffer_object_indices: None,
                descriptor_set_camera: None,
                descriptor_set_world: None,
                descriptor_writes_world: vec![],
                recreate_descriptor_sets: true,
                static_scene_changed: false,
            }
        });

        for resource in &mut self.resources {

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let uniform_buffer_camera = Buffer::new_sized::<CameraDataUBO>(memory_allocator.clone(), buffer_create_info, allocation_info)?;
            // let uniform_buffer_camera = Buffer::from_data(memory_allocator.clone(), buffer_create_info, allocation_info, camera_data)?;

            resource.buffer_camera_uniforms = Some(uniform_buffer_camera);

        }
        
        Ok(())
    }

    pub fn pre_render(&mut self, _ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        self.static_scene_changed = false;

        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if resource.recreate_descriptor_sets {
            resource.recreate_descriptor_sets = false;

            let graphics_pipeline = self.graphics_pipeline.as_ref().unwrap();
            Self::create_descriptor_sets(resource, graphics_pipeline, &engine.graphics)?;
        }

        self.check_changed_entities(&mut engine.scene);

        self.prepare_static_scene(&mut engine.scene);
        self.prepare_dynamic_scene(&mut engine.scene);

        self.update_buffer_capacity();

        let resource = &mut self.resources[frame_index];

        Self::map_object_data_buffer(resource, self.max_object_count as usize, engine.graphics.memory_allocator())?;
        Self::map_object_indices_buffer(resource, self.max_object_count as usize, engine.graphics.memory_allocator())?;

        let r = self.update_gpu_resources(engine);


        if let Err(r) = r {
            error!("Error writing buffers for frame: {} - Error was: {}", frame_index, r);
        }

        self.camera.update();

        // info!("Camera position: {:?}, Direction: {:?} - fov={}, aspect={}, far={}", self.camera.position(), self.camera.z_axis(), self.camera.fov(), self.camera.aspect_ratio(), self.camera.far_plane());



        Ok(())
    }

    pub fn render(&mut self, _ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        let viewport = engine.graphics.get_viewport();

        let resource = Self::curr_resource(&mut self.resources, engine);

        let uniform_buffer_camera = resource.buffer_camera_uniforms.as_ref().unwrap();

        match uniform_buffer_camera.write() {
            Ok(mut write) => self.camera.update_camera_buffer(&mut write),
            Err(err) => error!("Unable to write camera data: {err}")
        }

        let graphics_pipeline = self.graphics_pipeline.as_ref().unwrap();
        let descriptor_set_camera = resource.descriptor_set_camera.as_ref().unwrap();
        let descriptor_set_world = resource.descriptor_set_world.as_ref().unwrap();
        let pipeline_layout = graphics_pipeline.layout();

        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;
        cmd_buf.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![descriptor_set_camera.clone(), descriptor_set_world.clone()])?;

        if self.wireframe_mode == WireframeMode::Solid || self.wireframe_mode == WireframeMode::Both {
            let solid_graphics_pipeline = self.graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(solid_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene)?;
        }
        if self.wireframe_mode == WireframeMode::Wire || self.wireframe_mode == WireframeMode::Both {
            let wire_graphics_pipeline = self.wire_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(wire_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene)?;
        }

        Ok(())
    }

    fn check_changed_entities(&mut self, scene: &mut Scene) {

        let mut static_batch = vec![];
        let mut dynamic_batch = vec![];

        let mut query_added = scene.world.query_filtered::<(Entity, &mut RenderComponent<BaseVertex>), Added<RenderComponent<BaseVertex>>>();

        let a = query_added.iter(&scene.world).for_each(|(entity, render_component)| {

            match render_component.render_type {
                RenderType::Static => static_batch.push((entity, StaticRenderComponentMarker{})),
                RenderType::Dynamic => dynamic_batch.push((entity, DynamicRenderComponentMarker{}))
            };
        });

        if static_batch.len() > 0 {
            debug!("{} Static RenderComponent entities were added - change tick: {:?} to {:?}", static_batch.len(), scene.world.last_change_tick(), scene.world.change_tick());
            scene.world.insert_batch(static_batch);
            self.static_scene_changed = true;
            for resource in &mut self.resources {
                resource.static_scene_changed = true;
            }
        }

        if dynamic_batch.len() > 0 {
            debug!("{} Dynamic RenderComponent entities were added - change tick: {:?} to {:?}", dynamic_batch.len(), scene.world.last_change_tick(), scene.world.change_tick());
            scene.world.insert_batch(dynamic_batch);
        }

        let query_removed: Vec<Entity> = scene.world.removed::<RenderComponent<BaseVertex>>().collect();
        for entity in query_removed {
            scene.world.entity_mut(entity).remove::<(StaticRenderComponentMarker, DynamicRenderComponentMarker)>();
        }
    }

    fn prepare_static_scene(&mut self, scene: &mut Scene) {
        if !self.static_scene_changed {
            return; // Do nothing.
        }

        let mut query = scene.world.query_filtered::<(&mut RenderComponent<BaseVertex>, &Transform), With<StaticRenderComponentMarker>>();

        let start_index = 0;

        self.render_info.clear();
        self.object_data.clear();
        self.object_indices.clear();

        let mut index: u32 = 0;

        // let iter = query.par_iter(&scene.world);

        // TODO: par_iter
        query.iter(&scene.world).for_each(|(render_component, transform)| {

            let mut object_data = ObjectDataUBO::default();
            Self::update_object_date_transform(transform, &mut object_data);

            self.render_info.push(RenderInfo{
                mesh: render_component.mesh.clone(),
                index
            });

            self.object_data.push(object_data);
            self.object_indices.push(ObjectIndexUBO{ index });
            index += 1;
        });

        self.object_indices[start_index..].par_sort_unstable_by(|lhs_idx, rhs_idx| {
            let lhs = &self.render_info[lhs_idx.index as usize];
            let rhs = &self.render_info[rhs_idx.index as usize];

            lhs.mesh.cmp(&rhs.mesh)
        });

        self.static_object_count = index;
    }

    fn prepare_dynamic_scene(&mut self, scene: &mut Scene) {

        let mut query = scene.world.query_filtered::<(&mut RenderComponent<BaseVertex>, &Transform), With<DynamicRenderComponentMarker>>();

        let start_index = self.static_object_count as usize;

        self.render_info.truncate(start_index);
        self.object_data.truncate(start_index);
        self.object_indices.truncate(start_index);

        let mut index: u32 = self.static_object_count;

        // TODO: par_iter
        query.iter(&scene.world).for_each(|(render_component, transform)| {

            let mut object_data = ObjectDataUBO::default();
            Self::update_object_date_transform(transform, &mut object_data);

            self.render_info.push(RenderInfo{
                mesh: render_component.mesh.clone(),
                index
            });

            self.object_data.push(object_data);
            self.object_indices.push(ObjectIndexUBO{ index });
            index += 1;
        });

        self.object_indices[start_index..].par_sort_unstable_by(|lhs_idx, rhs_idx| {
            let lhs = &self.render_info[lhs_idx.index as usize];
            let rhs = &self.render_info[rhs_idx.index as usize];

            lhs.mesh.cmp(&rhs.mesh)
        });

        self.dynamic_object_count = index - self.static_object_count;
    }

    fn update_buffer_capacity(&mut self) {
        let num_objects = self.static_object_count + self.dynamic_object_count;

        if num_objects > self.max_object_count {
            // Grow the GPU buffers by 1.5x
            // TODO: tune this value - We don't want to over-allocate memory, and we don't want to resize the buffers too often.
            let growth_rate = 1.5;

            let prev_max_objects = self.max_object_count;
            self.max_object_count = (num_objects as f32 * growth_rate).ceil() as u32;
            debug!("SceneRenderer - Growing GPU buffers. Previous max objects: {prev_max_objects}, new max objects: {}", self.max_object_count);
        }
    }

    fn update_object_date_transform(transform: &Transform, object_data_buffer: &mut ObjectDataUBO) {
        transform.write_model_matrix(&mut object_data_buffer.model_matrix)
    }

    fn draw_scene(&self, cmd_buf: &mut PrimaryCommandBuffer, scene: &mut Scene) -> Result<()> {
        let mut draw_commands = vec![];

        let mut first_instance = 0;
        let mut curr_draw_command = None;

        for object_index in &self.object_indices {
            let render_info = &self.render_info[object_index.index as usize];

            if curr_draw_command.is_none() {
                curr_draw_command = Some(BatchedDrawCommand{
                    mesh: render_info.mesh.clone(),
                    instance_count: 0,
                    first_instance,
                });
            } else if curr_draw_command.as_ref().unwrap().mesh != render_info.mesh {
                first_instance += curr_draw_command.as_ref().unwrap().instance_count;
                draw_commands.push(curr_draw_command.as_ref().unwrap().clone());
                curr_draw_command = Some(BatchedDrawCommand{
                    mesh: render_info.mesh.clone(),
                    instance_count: 0,
                    first_instance,
                });
            }

            curr_draw_command.as_mut().unwrap().instance_count += 1;
        }

        if let Some(draw_command) = curr_draw_command && draw_command.instance_count > 0 {
            draw_commands.push(draw_command);
        }

        for draw_command in draw_commands {
            draw_command.mesh.draw(cmd_buf, draw_command.instance_count, draw_command.first_instance)?;
        }

        // let mut query = scene.world.query::<(Entity, &RenderComponent<BaseVertex>, &Transform)>();
        //
        // for (_entity, render_component, transform) in query.iter(&scene.world) {
        //     render_component.mesh.draw(cmd_buf, 1, 0)?;
        // }

        Ok(())
    }

    pub fn update_gpu_resources(&mut self, engine: &Engine) -> Result<()> {

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if !resource.descriptor_writes_world.is_empty() {
            let mut descriptor_set_world = resource.descriptor_set_world.as_mut().unwrap().clone();
            let writes = mem::take(&mut resource.descriptor_writes_world);
            // debug!("Writing world info descriptor sets: {writes:?}");

            unsafe { descriptor_set_world.update_by_ref(writes, []) }?

        }

        let static_count = self.static_object_count as usize;
        let dynamic_count = self.dynamic_object_count as usize;
        let static_begin = 0;
        let static_end = static_begin + static_count;
        let dynamic_begin = static_end;
        let dynamic_end = dynamic_begin + dynamic_count;

        if static_count > 0 || dynamic_count > 0 {

            {
                let buffer_object_data = resource.buffer_object_data.as_mut().unwrap();
                debug_assert!(self.object_data.len() <= buffer_object_data.len() as usize);
                let mut writer = buffer_object_data.write()?;

                if resource.static_scene_changed && static_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&self.object_data[static_begin..static_end]);
                }
                if dynamic_count > 0 {
                    writer[dynamic_begin..dynamic_end].copy_from_slice(&self.object_data[dynamic_begin..dynamic_end]);
                }
                // writer[..self.object_data.len()].copy_from_slice(self.object_data.as_slice());
            }


            {
                let buffer_object_indices = resource.buffer_object_indices.as_mut().unwrap();
                debug_assert!(self.object_indices.len() <= buffer_object_indices.len() as usize);
                let mut writer = buffer_object_indices.write()?;
                writer[..self.object_indices.len()].copy_from_slice(self.object_indices.as_slice());

                if resource.static_scene_changed && static_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&self.object_indices[static_begin..static_end]);
                }
                if dynamic_count > 0 {
                    writer[dynamic_begin..dynamic_end].copy_from_slice(&self.object_indices[dynamic_begin..dynamic_end]);
                }
            }
        }

        resource.static_scene_changed = false;

        Ok(())
    }

    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        debug!("SceneRenderer - Recreate swapchain");
        self.create_main_graphics_pipeline(engine)?;

        self.init_resources(engine)?;
        
        for resource in self.resources.iter_mut() {
            resource.recreate_descriptor_sets = true;
        }

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

        self.graphics_pipeline = Some(main_graphics_pipeline);
        self.wire_graphics_pipeline = Some(wire_graphics_pipeline);

        Ok(())
    }

    fn create_descriptor_sets(resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {

        debug!("SceneRenderer - create_descriptor_sets");

        let descriptor_set_allocator = graphics.descriptor_set_allocator();

        let buffer_camera_uniforms = resource.buffer_camera_uniforms.as_ref().unwrap();

        let pipeline_layout = graphics_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        // Camera info descriptor set
        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_writes = [WriteDescriptorSet::buffer(0, buffer_camera_uniforms.clone())];
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), descriptor_writes, [])?;
        resource.descriptor_set_camera = Some(descriptor_set);

        // World info descriptor set
        let descriptor_set_layout_index = 1;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), [], [])?;
        resource.descriptor_set_world = Some(descriptor_set);

        if let Some(buffer) = &resource.buffer_object_data {
            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(0, buffer.clone()));
        }
        if let Some(buffer) = &resource.buffer_object_indices {
            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(1, buffer.clone()));
        }

        Ok(())
    }

    fn map_object_data_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<StandardMemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<ObjectDataUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_object_data {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_object_data = None;
            }
        }

        if resource.buffer_object_data.is_none() {
            debug!("Allocating ObjectData GPU buffer for {max_object_count} objects");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_object_data = Buffer::new_slice::<ObjectDataUBO>(memory_allocator.clone(), buffer_create_info, allocation_info, max_object_count as DeviceSize)?;

            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(0, buffer_object_data.clone()));

            resource.buffer_object_data = Some(buffer_object_data);
        }

        Ok(())
    }

    fn map_object_indices_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<StandardMemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<ObjectIndexUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_object_indices {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_object_indices = None;
            }
        }

        if resource.buffer_object_indices.is_none() {
            debug!("Allocating ObjectIndices GPU buffer for {max_object_count} objects");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_object_indices = Buffer::new_slice::<ObjectIndexUBO>(memory_allocator.clone(), buffer_create_info, allocation_info, max_object_count as DeviceSize)?;

            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(1, buffer_object_indices.clone()));

            resource.buffer_object_indices = Some(buffer_object_indices);
        }

        Ok(())
    }

    fn curr_resource<'a>(resources: &'a mut [FrameResource], engine: &Engine) -> &'a mut FrameResource {
        Self::resource(resources, engine.graphics.current_frame_index())
    }

    fn resource(resources: &mut [FrameResource], index: usize) -> &mut FrameResource {
        // self.resources[index].as_mut().unwrap()
        &mut resources[index]
    }

    // pub fn add_mesh(&mut self, mesh: Mesh<BaseVertex>) {
    //     self.meshes.push(mesh);
    // }

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