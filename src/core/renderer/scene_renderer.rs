use crate::application::Ticker;
use crate::core::{set_vulkan_debug_name, util, CommandBuffer, CommandBufferImpl, Engine, FrameCompleteEvent, GraphicsManager, GraphicsPipelineBuilder, Material, Mesh, MeshData, MeshDataConfig, MeshPrimitiveType, RecreateSwapchainEvent, RenderCamera, RenderComponent, RenderType, Scene, StandardMemoryAllocator, Texture, Transform, VertexHasColour, VertexHasNormal, VertexHasPosition, VertexHasTexture};
use crate::{function_name, profile_scope_fn};
use anyhow::anyhow;
use anyhow::Result;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Added;
use bevy_ecs::query::With;
use foldhash::HashMap;
use glam::{Affine3A, Mat4, U16Vec2, U8Vec4, Vec2, Vec3, Vec3A, Vec4};
use log::{debug, error, info};
use rayon::slice::ParallelSliceMut;
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use std::any::Any;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineCreateFlags};
use vulkano::render_pass::Subpass;
use vulkano::DeviceSize;

#[derive(BufferContents, Vertex, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BaseVertex {
    #[format(R32G32B32_SFLOAT)]
    pub vs_position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub vs_normal: [f32; 3],
    #[format(R8G8B8A8_UNORM)]
    pub vs_colour: [u8; 4],
    #[format(R16G16_UNORM)]
    pub vs_texture: [u16; 2],
}

impl BaseVertex {
    pub fn new(position: Vec3, normal: Vec3, colour: U8Vec4, texture: U16Vec2) -> Self {
        BaseVertex {
            vs_position: [ position.x, position.y, position.z ],
            vs_normal: [ normal.x, normal.y, normal.z ],
            vs_colour: [ colour.x, colour.y, colour.z, colour.w ],
            vs_texture: [ texture.x, texture.y ]
        }
    }
}

impl Default for BaseVertex {
    fn default() -> Self {
        BaseVertex {
            vs_position: [0.0; 3],
            vs_normal: [0.0; 3],
            vs_colour: [255; 4],
            vs_texture: [0; 2],
        }
    }
}

impl VertexHasPosition<f32> for BaseVertex {
    fn position(&self) -> &[f32; 3] {
        &self.vs_position
    }

    fn set_position(&mut self, pos: [f32; 3]) {
        self.vs_position = pos;
    }

    fn transform_mat4(&mut self, transform: Mat4) {
        let p = transform.transform_point3a(Vec3A::new(self.vs_position[0], self.vs_position[1], self.vs_position[2]));
        self.vs_position = [p.x, p.y, p.z];
    }

    fn transform_affine(&mut self, transform: Affine3A) {
        let p = transform.transform_point3a(Vec3A::new(self.vs_position[0], self.vs_position[1], self.vs_position[2]));
        self.vs_position = [p.x, p.y, p.z];
    }
}

impl VertexHasNormal<f32> for BaseVertex {
    fn normal(&self) -> &[f32; 3] {
        &self.vs_normal
    }

    fn set_normal(&mut self, normal: [f32; 3]) {
        self.vs_normal = normal
    }
}

impl VertexHasColour<u8> for BaseVertex {
    fn colour(&self) -> &[u8; 4] {
        &self.vs_colour
    }

    fn set_colour(&mut self, colour: [u8; 4]) {
        self.vs_colour = colour;
    }
}

impl VertexHasTexture<u16> for BaseVertex {
    fn texture(&self) -> &[u16; 2] {
        &self.vs_texture
    }

    fn set_texture(&mut self, texture: [u16; 2]) {
        self.vs_texture = texture;
    }
}


#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct ObjectDataUBO {
    model_matrix: [f32; 16],
    material_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct DebugObjectDataUBO {
    model_matrix: [f32; 16],
    colour: [u8; 4],
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct ObjectIndexUBO {
    index: u32
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct MaterialUBO {
    texture_index: u32
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
    solid_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    wire_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    debug_lines_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    resources: Vec<FrameResource>,
    // render_camera: RenderCamera,
    // camera: Camera,
    // meshes: Vec<Mesh<BaseVertex>>, // temporary

    textures: Vec<Texture>,
    materials: Vec<Material>,
    textures_map: HashMap<u64, usize>,
    materials_map: HashMap<u64, usize>,
    textures_changed: bool,
    materials_changed: bool,

    render_info: Vec<RenderInfo>,
    object_data: Vec<ObjectDataUBO>,
    object_indices: Vec<ObjectIndexUBO>,
    material_data: Vec<MaterialUBO>,
    draw_commands: Vec<BatchedDrawCommand>,
    static_object_count: u32,
    dynamic_object_count: u32,
    max_object_count: u32,
    max_material_count: u32,
    static_scene_changed: bool,

    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_frame_complete: Option<ReaderId<FrameCompleteEvent>>,

    null_texture: Arc<ImageView>,
    default_sampler: Arc<Sampler>,

    debug_render_context: DebugRenderContext,
    max_debug_object_count: u32,

}

struct FrameResource {
    // buffer_camera_uniforms: Option<Subbuffer<CameraDataUBO>>,
    buffer_object_data: Option<Subbuffer<[ObjectDataUBO]>>,
    buffer_object_indices: Option<Subbuffer<[ObjectIndexUBO]>>,
    buffer_material_data: Option<Subbuffer<[MaterialUBO]>>,
    descriptor_set_camera: Option<Arc<DescriptorSet>>,
    descriptor_set_world: Option<Arc<DescriptorSet>>,
    descriptor_set_materials: Option<Arc<DescriptorSet>>,

    descriptor_writes_world: Vec<WriteDescriptorSet>,
    descriptor_writes_materials: Vec<WriteDescriptorSet>,

    recreate_descriptor_sets: bool,
    static_scene_changed: bool,
    textures_changed: bool,
    materials_changed: bool,
    camera_hash: u64,

    buffer_debug_object_data: Option<Subbuffer<[DebugObjectDataUBO]>>,
    buffer_debug_object_indices: Option<Subbuffer<[ObjectIndexUBO]>>,
    descriptor_set_debug_world: Option<Arc<DescriptorSet>>,
    descriptor_writes_debug_world: Vec<WriteDescriptorSet>,

    debug_render_info: Vec<RenderInfo>,
    debug_object_data: Vec<DebugObjectDataUBO>,
    debug_object_indices: Vec<ObjectIndexUBO>,
    debug_draw_commands: Vec<BatchedDrawCommand>,

    active_resources: Vec<Arc<dyn Any>>,
}

#[derive(Debug, Clone)]
pub struct DebugObjectInfo {
    transform: Transform,
    colour: U8Vec4,
}

#[derive(Default, Debug)]
pub struct DebugRenderContext {
    pub meshes: Vec<(Arc<Mesh<BaseVertex>>, DebugObjectInfo)>,
    immediate_meshes: Vec<(MeshData<BaseVertex>, DebugObjectInfo)>,
    mesh_builder: Option<MeshData<BaseVertex>>,
}

impl DebugRenderContext {
    pub fn add_mesh(&mut self, mesh: Arc<Mesh<BaseVertex>>, transform: Transform, colour: U8Vec4) {
        self.meshes.push((mesh, DebugObjectInfo { transform, colour }));
    }

    pub fn begin(&mut self, primitive_type: MeshPrimitiveType) -> &mut MeshData<BaseVertex> {
        assert_eq!(self.mesh_builder, None);
        self.mesh_builder = Some(MeshData::<BaseVertex>::new(MeshDataConfig::new(primitive_type)));
        self.mesh_builder.as_mut().unwrap()
    }

    pub fn end(&mut self) {
        assert_ne!(self.mesh_builder, None);
        let mesh_data = self.mesh_builder.take().unwrap();
        self.immediate_meshes.push((mesh_data, DebugObjectInfo { transform: Transform::default(), colour: U8Vec4::new(255, 255, 255, 255) }));
    }

    fn reset(&mut self) {
        self.meshes.clear();
        self.immediate_meshes.clear();
        self.mesh_builder = None;
    }

    fn object_count(&self) -> usize {
        self.meshes.len() + self.immediate_meshes.len()
    }
}

impl SceneRenderer {
    pub fn new(graphics: &GraphicsManager) -> Result<Self> {

        let null_texture = Self::create_null_texture(graphics)?;
        let default_sampler = Self::create_default_sampler(graphics.device().clone())?;

        let scene_renderer = SceneRenderer{

            solid_graphics_pipeline: None,
            wire_graphics_pipeline: None,
            debug_lines_graphics_pipeline: None,
            resources: vec![],
            // render_camera: RenderCamera::new(graphics),
            // camera: Camera::new(),

            textures: vec![],
            materials: vec![],
            textures_map: Default::default(),
            materials_map: Default::default(),
            textures_changed: false,
            materials_changed: false,
            render_info: vec![],
            object_data: vec![],
            object_indices: vec![],
            draw_commands: vec![],
            // meshes: vec![],

            material_data: vec![],
            static_object_count: 0,
            dynamic_object_count: 0,
            max_object_count: 100,
            max_material_count: 100,
            static_scene_changed: false,

            event_recreate_swapchain: None,
            event_frame_complete: None,

            null_texture,
            default_sampler,

            debug_render_context: Default::default(),
            max_debug_object_count: 0,

        };

        Ok(scene_renderer)
    }

    pub fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        self.event_frame_complete = Some(engine.graphics.event_bus().register::<FrameCompleteEvent>());
        Ok(())
    }


    pub fn init(&mut self, _engine: &mut Engine) -> Result<()> {
        let null_texture = Texture::new(self.null_texture.clone(), self.default_sampler.clone());
        self.register_texture(&null_texture);
        Ok(())
    }
    
    fn init_resources(&mut self, engine: &mut Engine) -> Result<()> {

        self.resources.resize_with(engine.graphics.max_concurrent_frames(), || {
            FrameResource{
                buffer_object_data: None,
                buffer_object_indices: None,
                buffer_material_data: None,
                descriptor_set_camera: None,
                descriptor_set_world: None,
                descriptor_set_materials: None,

                descriptor_writes_world: vec![],
                descriptor_writes_materials: vec![],
                
                recreate_descriptor_sets: true,
                static_scene_changed: false,
                textures_changed: false,
                materials_changed: false,
                camera_hash: 0,

                buffer_debug_object_data: None,
                buffer_debug_object_indices: None,
                descriptor_set_debug_world: None,
                descriptor_writes_debug_world: vec![],

                debug_render_info: vec![],
                debug_object_data: vec![],
                debug_object_indices: vec![],
                debug_draw_commands: vec![],

                active_resources: vec![]
            }
        });

        Ok(())
    }

    pub fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, _cmd_buf: &mut CommandBuffer) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);

        if ticker.time_since_last_dbg() >= ticker.debug_interval() {
            let mut query = engine.scene.ecs.query::<(&mut RenderComponent<BaseVertex>, &Transform)>();
            let len = query.iter(&engine.scene.ecs).len();
            debug!("{len} RenderComponent entities");
        }

        self.static_scene_changed = false;

        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }
        if let Some(e) = engine.graphics.event_bus().read_one_opt(&mut self.event_frame_complete) {
            self.on_frame_complete(engine, &e);
        }

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if resource.recreate_descriptor_sets {
            resource.recreate_descriptor_sets = false;

            Self::create_main_descriptor_sets(resource, self.solid_graphics_pipeline.as_ref().unwrap(), &engine.graphics)?;
            Self::create_debug_descriptor_sets(resource, self.debug_lines_graphics_pipeline.as_ref().unwrap(), &engine.graphics)?;
        }

        self.check_changed_entities(&mut engine.scene);

        self.prepare_static_scene(&mut engine.scene);
        self.prepare_dynamic_scene(&mut engine.scene);

        self.check_changed_materials(&mut engine.scene);

        self.update_buffer_capacities();

        let resource = &mut self.resources[frame_index];

        Self::map_object_data_buffer(resource, self.max_object_count as usize, engine.graphics.memory_allocator())?;
        Self::map_object_indices_buffer(resource, self.max_object_count as usize, engine.graphics.memory_allocator())?;
        Self::map_materials_buffer(resource, self.max_material_count as usize, engine.graphics.memory_allocator())?;

        if let Err(r) = self.update_gpu_resources(engine) {
            error!("Error writing buffers for frame: {frame_index} - Error was: {r}");
        }


        // self.camera.update();

        // info!("Camera position: {:?}, Direction: {:?} - fov={}, aspect={}, far={}", self.camera.position(), self.camera.z_axis(), self.camera.fov(), self.camera.aspect_ratio(), self.camera.far_plane());

        Ok(())
    }

    pub fn render(&mut self, camera: &RenderCamera, _ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()> {

        profile_scope_fn!(&engine.frame_profiler);

        let viewport = engine.graphics.get_viewport();

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if Self::check_changed_camera(camera, resource, frame_index) {
            // debug!("=========== XX UPDATING CAMERA DESCRIPTOR SETS XX ===========");
            Self::create_camera_descriptor_sets(&camera, resource, self.solid_graphics_pipeline.as_ref().unwrap(), &engine.graphics)?;
            resource.camera_hash = camera.gpu_resource_hash(frame_index);
        }

        // let uniform_buffer_camera = resource.buffer_camera_uniforms.as_ref().unwrap();
        //
        // match uniform_buffer_camera.write() {
        //     Ok(mut write) => self.camera.update_camera_buffer(&mut write),
        //     Err(err) => error!("Unable to write camera data: {err}")
        // }

        let graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
        let descriptor_set_camera = resource.descriptor_set_camera.as_ref().unwrap();
        let descriptor_set_world = resource.descriptor_set_world.as_ref().unwrap();
        let descriptor_set_materials = resource.descriptor_set_materials.as_ref().unwrap();
        let pipeline_layout = graphics_pipeline.layout();

        cmd_buf.set_viewport(0, [viewport].into_iter().collect())?;
        cmd_buf.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![descriptor_set_camera.clone(), descriptor_set_world.clone(), descriptor_set_materials.clone()])?;

        self.draw_commands.clear();
        Self::build_batched_draw_commands(&self.object_indices, &self.render_info, &mut self.draw_commands);

        if engine.wireframe_mode().is_solid() {
            let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(solid_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene, engine.graphics.current_frame_index())?;
        }
        if engine.wireframe_mode().is_wire() {
            let wire_graphics_pipeline = self.wire_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(wire_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene, engine.graphics.current_frame_index())?;
        }

        self.render_debug(engine, cmd_buf)?;

        Ok(())
    }

    pub fn render_debug(&mut self, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()> {

        profile_scope_fn!(&engine.frame_profiler);

        let debug_object_count = self.debug_render_context.object_count();
        if debug_object_count == 0 {
            return Ok(())
        }

        self.max_debug_object_count = util::grow_capacity(self.max_debug_object_count, debug_object_count as u32, 1.5);

        // debug!("{} debug meshes", debug_object_count);

        let allocator = engine.graphics.memory_allocator();

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        if !self.debug_render_context.immediate_meshes.is_empty() {

            while let Some((mesh_data, object_info)) = self.debug_render_context.immediate_meshes.pop() {
                let mesh = mesh_data.build_mesh(allocator.clone())?;
                self.debug_render_context.add_mesh(Arc::new(mesh), object_info.transform, object_info.colour);
            }

            // let staging_buffer = GraphicsManager::create_staging_subbuffer::<u8>(allocator.clone(), staging_buffer_size)?;
            // resource.active_resources.push(staging_buffer.buffer().clone());
            //
            // let mut curr_staging_buffer = Some(staging_buffer);
            //
            // let mut cmd_buf = engine.graphics.begin_transfer_commands()?;
            //
            // while let Some((mesh_data, object_info)) = self.debug_render_context.immediate_meshes.pop() {
            //     let staging_buffer = util::chop_buffer_at(&mut curr_staging_buffer, mesh_data.get_required_staging_buffer_size()).unwrap();
            //     let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;
            //     self.debug_render_context.add_mesh(Arc::new(mesh), object_info.transform, object_info.colour);
            // }
            //
            // engine.graphics.submit_transfer_commands(cmd_buf)?
            //     .wait(None)?;
        }

        let debug_lines_pipeline = self.debug_lines_graphics_pipeline.as_ref().unwrap();
        cmd_buf.bind_pipeline_graphics(debug_lines_pipeline.clone())?;

        Self::map_debug_object_data_buffer(resource, self.max_debug_object_count as usize, allocator.clone())?;
        Self::map_debug_object_indices_buffer(resource, self.max_debug_object_count as usize, allocator.clone())?;

        resource.debug_render_info.clear();
        resource.debug_object_data.clear();
        resource.debug_object_indices.clear();
        resource.debug_draw_commands.clear();

        let mut index = 0;

        for (mesh, object_info) in self.debug_render_context.meshes.iter() {
            let mut obj = DebugObjectDataUBO::default();
            object_info.transform.write_model_matrix(&mut obj.model_matrix);
            object_info.colour.write_to_slice(&mut obj.colour);

            resource.debug_render_info.push(RenderInfo{
                mesh: mesh.clone(),
                index
            });

            resource.debug_object_data.push(obj);
            resource.debug_object_indices.push(ObjectIndexUBO{ index });

            index += 1;
        }

        Self::update_debug_object_data_gpu_resources(resource)?;

        Self::build_batched_draw_commands(&resource.debug_object_indices, &resource.debug_render_info, &mut resource.debug_draw_commands);

        let debug_lines_graphics_pipeline = self.debug_lines_graphics_pipeline.as_ref().unwrap();
        let descriptor_set_camera = resource.descriptor_set_camera.as_ref().unwrap();
        let descriptor_set_debug_world = resource.descriptor_set_debug_world.as_ref().unwrap();
        let pipeline_layout = debug_lines_graphics_pipeline.layout();

        cmd_buf.bind_descriptor_sets(PipelineBindPoint::Graphics, pipeline_layout.clone(), 0, vec![descriptor_set_camera.clone(), descriptor_set_debug_world.clone()])?;
        cmd_buf.bind_pipeline_graphics(debug_lines_graphics_pipeline.clone())?;

        for draw_command in resource.debug_draw_commands.iter() {
            draw_command.mesh.draw(cmd_buf, draw_command.instance_count, draw_command.first_instance)?;
            resource.active_resources.push(draw_command.mesh.clone());
        }

        self.debug_render_context.reset();
        Ok(())
    }

    fn check_changed_camera(camera: &RenderCamera, resource: &FrameResource, frame_index: usize) -> bool {
        camera.gpu_resource_hash(frame_index) != resource.camera_hash
    }

    fn check_changed_entities(&mut self, scene: &mut Scene) {

        let mut static_scene_changed = false;
        
        let mut static_batch = vec![];
        let mut dynamic_batch = vec![];

        let mut query_added = scene.ecs.query_filtered::<(Entity, &mut RenderComponent<BaseVertex>), Added<RenderComponent<BaseVertex>>>();

        query_added.iter(&scene.ecs).for_each(|(entity, render_component)| {
            match render_component.render_type {
                RenderType::Static => static_batch.push((entity, StaticRenderComponentMarker {})),
                RenderType::Dynamic => dynamic_batch.push((entity, DynamicRenderComponentMarker {}))
            };
        });

        if !static_batch.is_empty() {
            // debug!("{} Static RenderComponent entities were added - change tick: {:?} to {:?}", static_batch.len(), scene.ecs.last_change_tick(), scene.ecs.change_tick());
            scene.ecs.insert_batch(static_batch);
            static_scene_changed = true;
        }

        if !dynamic_batch.is_empty() {
            // debug!("{} Dynamic RenderComponent entities were added - change tick: {:?} to {:?}", dynamic_batch.len(), scene.ecs.last_change_tick(), scene.ecs.change_tick());
            scene.ecs.insert_batch(dynamic_batch);
        }

        let query_removed: Vec<Entity> = scene.ecs.removed::<RenderComponent<BaseVertex>>().collect();
        
        for entity in query_removed {
            if !static_scene_changed && scene.ecs.get::<StaticRenderComponentMarker>(entity).is_some() {
                static_scene_changed = true;
            }
            _ = scene.ecs.get_entity_mut(entity).map(|mut entity| {
                entity.remove::<(StaticRenderComponentMarker, DynamicRenderComponentMarker)>();
            });
                                                 
            // scene.ecs.entity_mut(entity).remove::<(StaticRenderComponentMarker, DynamicRenderComponentMarker)>();
        }
        
        if static_scene_changed {
            self.static_scene_changed = true;
            for resource in &mut self.resources {
                resource.static_scene_changed = true;
            }
        }
    }

    fn check_changed_materials(&mut self, _scene: &mut Scene) {
        if self.textures_changed {
            debug!("Textures changed");
            for resource in &mut self.resources {
                resource.textures_changed = true;
            }
        }
        if self.materials_changed {
            debug!("Materials changed");
            for resource in &mut self.resources {
                resource.materials_changed = true;
            }
        }
        self.textures_changed = false;
        self.materials_changed = false;
    }

    fn prepare_static_scene(&mut self, scene: &mut Scene) {
        if !self.static_scene_changed {
            return; // Do nothing.
        }

        let mut query = scene.ecs.query_filtered::<(&mut RenderComponent<BaseVertex>, &Transform), With<StaticRenderComponentMarker>>();

        let start_index = 0;

        self.render_info.clear();
        self.object_data.clear();
        self.object_indices.clear();

        let mut index: u32 = 0;

        // let iter = query.par_iter(&scene.world);

        // TODO: par_iter
        query.iter(&scene.ecs).for_each(|(render_component, transform)| {

            if render_component.mesh().is_none() {
                return; // continue
            }
            
            self.prepare_scene_object(index, render_component, transform);
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

        let mut query = scene.ecs.query_filtered::<(&mut RenderComponent<BaseVertex>, &Transform), With<DynamicRenderComponentMarker>>();

        let start_index = self.static_object_count as usize;

        self.render_info.truncate(start_index);
        self.object_data.truncate(start_index);
        self.object_indices.truncate(start_index);

        let mut index: u32 = self.static_object_count;

        // TODO: par_iter
        query.iter(&scene.ecs).for_each(|(render_component, transform)| {
            
            if render_component.mesh().is_none() {
                return; // continue
            }
            
            self.prepare_scene_object(index, render_component, transform);
            index += 1;
        });

        self.object_indices[start_index..].par_sort_unstable_by(|lhs_idx, rhs_idx| {
            let lhs = &self.render_info[lhs_idx.index as usize];
            let rhs = &self.render_info[rhs_idx.index as usize];

            lhs.mesh.cmp(&rhs.mesh)
        });

        self.dynamic_object_count = index - self.static_object_count;
    }

    fn prepare_scene_object(&mut self, index: u32, render_component: &RenderComponent<BaseVertex>, transform: &Transform) {
        let mut object_data = ObjectDataUBO::default();
        Self::update_object_date_transform(transform, &mut object_data);

        if let Some(material) = render_component.material() {
            self.register_material(material);
        }

        self.render_info.push(RenderInfo{
            mesh: render_component.mesh().unwrap().clone(),
            index
        });

        self.object_data.push(object_data);
        self.object_indices.push(ObjectIndexUBO{ index });
    }

    fn update_buffer_capacities(&mut self) {
        let num_objects = self.static_object_count + self.dynamic_object_count;

        // Grow the GPU buffers by 1.5x
        // TODO: tune this value - We don't want to over-allocate memory, and we don't want to resize the buffers too often.
        let growth_rate = 1.5;

        if num_objects > self.max_object_count {
            let prev_max_objects = self.max_object_count;
            self.max_object_count = util::grow_capacity(self.max_object_count, num_objects, growth_rate);
            debug!("SceneRenderer - Growing ObjectData GPU buffers. Previous max objects: {prev_max_objects}, new max objects: {}", self.max_object_count);
        }

        let num_materials = self.materials.len() as u32;

        if num_materials > self.max_material_count {
            let prev_max_materials = self.max_material_count;
            self.max_material_count = util::grow_capacity(self.max_material_count, num_objects, growth_rate);
            debug!("SceneRenderer - Growing Material GPU buffers. Previous max materials: {prev_max_materials}, new max materials: {}", self.max_material_count);
        }
    }

    fn update_object_date_transform(transform: &Transform, object_data_buffer: &mut ObjectDataUBO) {
        transform.write_model_matrix(&mut object_data_buffer.model_matrix)
    }

    fn draw_scene(&mut self, cmd_buf: &mut CommandBuffer, _scene: &mut Scene, frame_index: usize) -> Result<()> {

        let resource = &mut self.resources[frame_index];

        for draw_command in self.draw_commands.iter() {
            draw_command.mesh.draw(cmd_buf, draw_command.instance_count, draw_command.first_instance)?;
            resource.active_resources.push(draw_command.mesh.clone());
        }

        // let mut query = scene.world.query::<(Entity, &RenderComponent<BaseVertex>, &Transform)>();
        //
        // for (_entity, render_component, transform) in query.iter(&scene.world) {
        //     render_component.mesh.draw(cmd_buf, 1, 0)?;
        // }

        Ok(())
    }

    fn build_batched_draw_commands(object_indices: &[ObjectIndexUBO], render_info: &[RenderInfo], out_draw_commands: &mut Vec<BatchedDrawCommand>) {
        // let mut draw_commands = vec![];

        let mut first_instance = 0;
        let mut curr_draw_command = None;

        for object_index in object_indices {
            let render_info = &render_info[object_index.index as usize];

            if curr_draw_command.is_none() {
                curr_draw_command = Some(BatchedDrawCommand{
                    mesh: render_info.mesh.clone(),
                    instance_count: 0,
                    first_instance,
                });
            } else if curr_draw_command.as_ref().unwrap().mesh != render_info.mesh {
                first_instance += curr_draw_command.as_ref().unwrap().instance_count;
                out_draw_commands.push(curr_draw_command.as_ref().unwrap().clone());
                curr_draw_command = Some(BatchedDrawCommand{
                    mesh: render_info.mesh.clone(),
                    instance_count: 0,
                    first_instance,
                });
            }

            curr_draw_command.as_mut().unwrap().instance_count += 1;
        }

        if let Some(draw_command) = curr_draw_command && draw_command.instance_count > 0 {
            out_draw_commands.push(draw_command);
        }
    }

    fn update_gpu_resources(&mut self, engine: &Engine) -> Result<()> {

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        Self::update_object_data_gpu_resources(self.static_object_count as usize, self.dynamic_object_count as usize, &self.object_data, &self.object_indices, resource)?;

        Self::update_material_data_gpu_resources(&self.textures, &self.material_data, self.null_texture.clone(), self.default_sampler.clone(), resource)?;

        Ok(())
    }

    fn update_object_data_gpu_resources(static_count: usize, dynamic_count: usize, object_data: &[ObjectDataUBO], object_indices: &[ObjectIndexUBO], resource: &mut FrameResource) -> Result<()> {

        if !resource.descriptor_writes_world.is_empty() {
            let descriptor_set_world = resource.descriptor_set_world.as_ref().unwrap().clone();
            let writes = mem::take(&mut resource.descriptor_writes_world);

            unsafe { descriptor_set_world.update_by_ref(writes, []) }?
        }

        let static_begin = 0;
        let static_end = static_begin + static_count;
        let dynamic_begin = static_end;
        let dynamic_end = dynamic_begin + dynamic_count;

        if static_count > 0 || dynamic_count > 0 {

            {
                let buffer_object_data = resource.buffer_object_data.as_mut().unwrap();
                debug_assert!(object_data.len() <= buffer_object_data.len() as usize);
                let mut writer = buffer_object_data.write()?;

                if resource.static_scene_changed && static_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&object_data[static_begin..static_end]);
                }
                if dynamic_count > 0 {
                    writer[dynamic_begin..dynamic_end].copy_from_slice(&object_data[dynamic_begin..dynamic_end]);
                }
            }


            {
                let buffer_object_indices = resource.buffer_object_indices.as_mut().unwrap();
                debug_assert!(object_indices.len() <= buffer_object_indices.len() as usize);
                let mut writer = buffer_object_indices.write()?;

                if resource.static_scene_changed && static_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&object_indices[static_begin..static_end]);
                }
                if dynamic_count > 0 {
                    writer[dynamic_begin..dynamic_end].copy_from_slice(&object_indices[dynamic_begin..dynamic_end]);
                }
            }
        }

        resource.static_scene_changed = false;

        Ok(())
    }

    fn update_material_data_gpu_resources(textures: &[Texture], material_data: &[MaterialUBO], null_texture: Arc<ImageView>, default_sampler: Arc<Sampler>, resource: &mut FrameResource) -> Result<()> {

        if resource.textures_changed {
            debug!("Textures changed: Writing 0..{} textures", textures.len());
            let descriptor_set_materials = resource.descriptor_set_materials.as_ref().unwrap().clone();
            let binding = 0;
            let binding_info = descriptor_set_materials.layout().bindings().get(&binding).unwrap();
            let descriptor_count = binding_info.descriptor_count as usize;

            let mut elements: Vec<(Arc<ImageView>, Arc<Sampler>)> = textures.iter().map(|tex| (tex.image_view().clone(), tex.sampler().clone())).collect();
            elements.resize_with(descriptor_count, || (null_texture.clone(), default_sampler.clone()));

            let write = WriteDescriptorSet::image_view_sampler_array(binding, 0, elements);
            resource.descriptor_writes_materials.push(write);
        }

        if !resource.descriptor_writes_materials.is_empty() {
            let descriptor_set_materials = resource.descriptor_set_materials.as_ref().unwrap().clone();
            let writes = mem::take(&mut resource.descriptor_writes_materials);

            unsafe { descriptor_set_materials.update_by_ref(writes, []) }?
        }

        if resource.materials_changed {

            let idx_begin = 0;
            let idx_end = material_data.len();

            debug!("Material data changed: Uploading {idx_begin}..{idx_end} materials");

            let buffer_material_data = resource.buffer_material_data.as_mut().unwrap();
            debug_assert!(material_data.len() <= buffer_material_data.len() as usize);
            let mut writer = buffer_material_data.write()?;

            writer[idx_begin..idx_end].copy_from_slice(&material_data[idx_begin..idx_end]);
        }

        resource.textures_changed = false;
        resource.materials_changed = false;

        Ok(())
    }

    fn update_debug_object_data_gpu_resources(resource: &mut FrameResource) -> Result<()> {

        if !resource.descriptor_writes_debug_world.is_empty() {
            let descriptor_set_debug_world = resource.descriptor_set_debug_world.as_ref().unwrap().clone();
            let writes = mem::take(&mut resource.descriptor_writes_debug_world);

            unsafe { descriptor_set_debug_world.update_by_ref(writes, []) }?
        }

        let buffer_debug_object_data = resource.buffer_debug_object_data.as_mut().unwrap();

        let idx_begin = 0;
        let idx_end = resource.debug_object_data.len();

        debug_assert!(resource.debug_object_data.len() <= buffer_debug_object_data.len() as usize);
        let mut writer = buffer_debug_object_data.write()?;
        writer[idx_begin .. idx_end].copy_from_slice(&resource.debug_object_data);

        let buffer_debug_object_indices = resource.buffer_debug_object_indices.as_mut().unwrap();
        debug_assert!(resource.debug_object_indices.len() <= buffer_debug_object_indices.len() as usize);
        let mut writer = buffer_debug_object_indices.write()?;
        writer[idx_begin .. idx_end].copy_from_slice(&resource.debug_object_indices);

        Ok(())
    }

    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        debug!("SceneRenderer - Recreate swapchain");
        self.create_main_graphics_pipeline(engine)?;
        self.create_debug_graphics_pipeline(engine)?;

        self.init_resources(engine)?;
        
        for resource in self.resources.iter_mut() {
            resource.recreate_descriptor_sets = true;
        }

        Ok(())
    }

    fn on_frame_complete(&mut self, engine: &mut Engine, event: &FrameCompleteEvent) {
        let resource = &mut self.resources[event.frame_index];
        // if !resource.active_resources.is_empty() {
        //     let discard_count: usize = resource.active_resources.iter().map(|r| (Arc::strong_count(r) == 1) as usize).sum();
        //     if discard_count > 0 {
        //         debug!("Discarding {} resources", discard_count);
        //     }
        // }

        // We simply stop holding a reference to the resource, it will be cleaned up when dropped.
        resource.active_resources.clear();
    }

    fn create_main_graphics_pipeline(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize SceneRenderer GraphicsPipeline");
        let device = engine.graphics.device();
        let render_pass = engine.graphics.render_pass();

        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/world_solid.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "world_solid.glsl::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
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

    fn create_debug_graphics_pipeline(&mut self, engine: &Engine) -> Result<()> {
        info!("Initialize debug GraphicsPipeline");

        let device = engine.graphics.device();
        let render_pass = engine.graphics.render_pass();

        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/debug.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "debug.glsl::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
        options.add_macro_definition("FRAGMENT_SHADER_MODULE", None);
        let fs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "debug.glsl::frag(solid)", "main", ShaderKind::Fragment, Some(&options))?;

        let subpass_type = Subpass::from(render_pass.clone(), 0)
            .ok_or_else(|| anyhow!("Failed to get subpass info for provided RenderPass"))?
            .into();

        let entry_points = HashMap::from_iter([(0, "main"), (1, "main") ]);

        let debug_lines_graphics_pipeline = GraphicsPipelineBuilder::new(subpass_type, vec![vs.clone(), fs.clone()], entry_points.clone())
            .add_flags(PipelineCreateFlags::ALLOW_DERIVATIVES)
            .set_dynamic_states(vec![ DynamicState::Viewport ])
            .set_input_assembly_state(InputAssemblyState{
                topology: PrimitiveTopology::LineList,
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
            .set_depth_stencil_state(DepthStencilState{
                depth: Some(DepthState{ write_enable: true, compare_op: CompareOp::Less }),
                // depth_bounds: Some(0.0..=1.0),
                ..Default::default()
            })
            .set_color_blend_state(ColorBlendState{
                attachments: vec![ColorBlendAttachmentState{
                    blend: Some(AttachmentBlend::alpha()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: true,
                    ..Default::default()
                } ],
                ..Default::default()
            })
            .build_pipeline::<BaseVertex>(device.clone())?;

        self.debug_lines_graphics_pipeline = Some(debug_lines_graphics_pipeline);

        Ok(())
    }

    fn create_camera_descriptor_sets(camera: &RenderCamera, resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {

        let descriptor_set = camera.create_camera_descriptor_sets(0, 0, graphics_pipeline, graphics)?;
        resource.descriptor_set_camera = Some(descriptor_set);

        Ok(())
    }

    fn create_main_descriptor_sets(resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {
        debug!("SceneRenderer - create_main_descriptor_sets");

        let descriptor_set_allocator = graphics.descriptor_set_allocator();

        // let buffer_camera_uniforms = resource.buffer_camera_uniforms.as_ref().unwrap();

        let pipeline_layout = graphics_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();


        // World info descriptor set
        let descriptor_set_layout_index = 1;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), [], [])?;
        resource.descriptor_set_world = Some(descriptor_set);

        // Material info descriptor set
        let descriptor_set_layout_index = 2;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), [], [])?;
        resource.descriptor_set_materials = Some(descriptor_set);

        if let Some(buffer) = &resource.buffer_object_data {
            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(0, buffer.clone()));
        }
        if let Some(buffer) = &resource.buffer_object_indices {
            resource.descriptor_writes_world.push(WriteDescriptorSet::buffer(1, buffer.clone()));
        }
        if let Some(buffer) = &resource.buffer_material_data {
            resource.descriptor_writes_materials.push(WriteDescriptorSet::buffer(1, buffer.clone()));
        }
        resource.textures_changed = true;
        resource.materials_changed = true;

        Ok(())
    }

    fn create_debug_descriptor_sets(resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {
        debug!("SceneRenderer - create_debug_descriptor_sets");

        let descriptor_set_allocator = graphics.descriptor_set_allocator();

        let pipeline_layout = graphics_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        // World info descriptor set
        let descriptor_set_layout_index = 1;
        let descriptor_set_layout = descriptor_set_layouts.get(descriptor_set_layout_index).unwrap();
        let descriptor_set = DescriptorSet::new(descriptor_set_allocator.clone(), descriptor_set_layout.clone(), [], [])?;
        resource.descriptor_set_debug_world = Some(descriptor_set);

        if let Some(buffer) = &resource.buffer_debug_object_data {
            resource.descriptor_writes_debug_world.push(WriteDescriptorSet::buffer(0, buffer.clone()));
        }
        if let Some(buffer) = &resource.buffer_debug_object_indices {
            resource.descriptor_writes_debug_world.push(WriteDescriptorSet::buffer(1, buffer.clone()));
        }

        Ok(())
    }

    fn map_object_data_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<ObjectDataUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_object_data {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_object_data = None;
            }
        }

        if resource.buffer_object_data.is_none() {
            debug!("Allocating ObjectData GPU buffer for {max_object_count} objects ({max_buffer_size} bytes)");

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

    fn map_object_indices_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<ObjectIndexUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_object_indices {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_object_indices = None;
            }
        }

        if resource.buffer_object_indices.is_none() {
            debug!("Allocating ObjectIndices GPU buffer for {max_object_count} objects ({max_buffer_size} bytes)");

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

    fn map_materials_buffer(resource: &mut FrameResource, max_material_count: usize, memory_allocator: Arc<StandardMemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<MaterialUBO>() * max_material_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_material_data {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_material_data = None;
            }
        }

        if resource.buffer_material_data.is_none() {
            debug!("Allocating Materials GPU buffer for {max_material_count} materials ({max_buffer_size} bytes)");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_materials = Buffer::new_slice::<MaterialUBO>(memory_allocator.clone(), buffer_create_info, allocation_info, max_material_count as DeviceSize)?;

            resource.descriptor_writes_materials.push(WriteDescriptorSet::buffer(1, buffer_materials.clone()));

            resource.buffer_material_data = Some(buffer_materials);
        }

        Ok(())
    }

    fn map_debug_object_data_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<DebugObjectDataUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_debug_object_data {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_debug_object_data = None;
            }
        }

        if resource.buffer_debug_object_data.is_none() {
            debug!("Allocating Debug ObjectData GPU buffer for {max_object_count} objects ({max_buffer_size} bytes)");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_debug_object_data = Buffer::new_slice::<DebugObjectDataUBO>(memory_allocator.clone(), buffer_create_info, allocation_info, max_object_count as DeviceSize)?;

            resource.descriptor_writes_debug_world.push(WriteDescriptorSet::buffer(0, buffer_debug_object_data.clone()));

            resource.buffer_debug_object_data = Some(buffer_debug_object_data);
        }

        Ok(())
    }

    fn map_debug_object_indices_buffer(resource: &mut FrameResource, max_object_count: usize, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<ObjectIndexUBO>() * max_object_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_debug_object_indices {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_debug_object_indices = None;
            }
        }

        if resource.buffer_debug_object_indices.is_none() {
            debug!("Allocating Debug ObjectIndices GPU buffer for {max_object_count} objects ({max_buffer_size} bytes)");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_debug_object_indices = Buffer::new_slice::<ObjectIndexUBO>(memory_allocator.clone(), buffer_create_info, allocation_info, max_object_count as DeviceSize)?;

            resource.descriptor_writes_debug_world.push(WriteDescriptorSet::buffer(1, buffer_debug_object_indices.clone()));

            resource.buffer_debug_object_indices = Some(buffer_debug_object_indices);
        }

        Ok(())
    }

    pub fn register_texture(&mut self, texture: &Texture) -> usize {

        let id = texture.resource_id();
        let index = self.textures_map.entry(id).or_insert_with(|| {
            let index = self.textures.len();
            debug!("Registering texture: {index}");
            self.textures.push(texture.clone());
            self.textures_changed = true;
            index
        });

        *index
    }

    pub fn unregister_texture_index(&mut self, texture_index: usize) -> bool {
        if texture_index > self.textures.len() {
            return false;
        }

        // TODO: we need to remove the texture from the array, and ensure index references for remaining textures are correct.
        // let texture = &self.textures[texture_index];
        // self.textures.swap_remove()
        // true
        false
    }

    pub fn register_material(&mut self, material: &Material) -> usize {

        let texture_idx = self.register_texture(material.texture());

        let id = material.resource_id();
        let index = self.materials_map.entry(id).or_insert_with(|| {
            let index = self.materials.len();
            debug!("Registering material: {index}");
            self.materials.push(material.clone());
            self.material_data.push(MaterialUBO{
                texture_index: texture_idx as u32,
            });
            self.materials_changed = true;
            index
        });

        *index
    }

    fn create_null_texture(graphics: &GraphicsManager) -> Result<Arc<ImageView>>{
        let allocator = graphics.memory_allocator();

        let width = 2;
        let height = 2;
        
        const PIXEL_0: [u8; 4] = [0, 0, 0, 255];
        const PIXEL_1: [u8; 4] = [255, 0, 255, 255];
        
        let mut data = vec![128; (4 * width * height) as usize];
        data[0..4].clone_from_slice(&PIXEL_0);
        data[4..8].clone_from_slice(&PIXEL_1);
        data[8..12].clone_from_slice(&PIXEL_1);
        data[12..16].clone_from_slice(&PIXEL_0);
        
        let mut cmd_buf = graphics.begin_transfer_commands()?;
        
        let staging_buffer = GraphicsManager::create_staging_subbuffer::<u8>(allocator.clone(), data.len() as DeviceSize)?;
        set_vulkan_debug_name(staging_buffer.buffer(), Some("SceneRenderer-CreateNullTexture-StagingBuffer"))?;
        let image_view = Texture::create_image_view_2d(allocator, width, height, Format::R8G8B8A8_UNORM, ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST, Some("SceneRenderer-NullTexture"))?;

        Texture::load_image_from_data_staged(&mut cmd_buf, &staging_buffer, &data, image_view.image().clone())?;

        graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;
        
        Ok(image_view)
    }

    fn create_default_sampler(device: Arc<Device>) -> Result<Arc<Sampler>> {
        Texture::create_default_sampler(device)
    }

    // pub fn add_mesh(&mut self, mesh: Mesh<BaseVertex>) {
    //     self.meshes.push(mesh);
    // }

    pub fn debug_render_context(&mut self) -> &mut DebugRenderContext {
        &mut self.debug_render_context
    }
}