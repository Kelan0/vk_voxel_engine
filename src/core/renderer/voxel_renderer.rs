use crate::application::Ticker;
use crate::core::world::{closest_point_on_chunk, get_chunk_pos_for_world_pos, VoxelChunkData, VoxelWorld};
use crate::core::VertexHasPosition;
use crate::core::VertexHasNormal;
use crate::core::VertexHasColour;
use crate::core::Transform;
use crate::core::Texture;
use crate::core::StandardMemoryAllocator;
use crate::core::Scene;
use crate::core::RenderType;
use crate::core::RenderCamera;
use crate::core::RecreateSwapchainEvent;
use crate::core::MeshPrimitiveType;
use crate::core::MeshData;
use crate::core::MeshConfiguration;
use crate::core::Mesh;
use crate::core::Material;
use crate::core::GraphicsPipelineBuilder;
use crate::core::GraphicsManager;
use crate::core::FrameCompleteEvent;
use crate::core::Engine;
use crate::core::CommandBufferImpl;
use crate::core::CommandBuffer;
use crate::core::AxisDirection;
use crate::core::AxisAlignedBoundingBox;
use crate::core::util;
use crate::core::set_vulkan_debug_name;
use crate::core::VertexHasTexture;
use crate::{function_name, profile_scope_fn};
use anyhow::anyhow;
use anyhow::Result;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::prelude::Added;
use foldhash::{HashMap, HashMapExt, HashSet, HashSetExt};
use glam::{DVec3, IVec2, IVec3};
use log::{debug, error, info};
use shaderc::{CompileOptions, ShaderKind};
use shrev::ReaderId;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::{array, mem};
use std::hash::{DefaultHasher, Hash, Hasher};
use egui::{Context};
use rayon::slice::ParallelSliceMut;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, IndexType, Subbuffer};
use vulkano::command_buffer::DrawIndexedIndirectCommand;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::image::ImageUsage;
use vulkano::memory::allocator::{align_down, align_up, AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineCreateFlags};
use vulkano::render_pass::Subpass;
use vulkano::DeviceSize;
use vulkano::memory::DeviceAlignment;
use crate::core::MeshBufferOption::{AllocateNew, UseExisting};

#[derive(BufferContents, Vertex, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct VoxelVertex {
    #[format(R32_UINT)]
    pub vs_data: u32, // XXXXXYYYYYZZZZZNNN // X[5] Y[5] Z[5] Norm[3]
}

impl VoxelVertex {
    const MASK_X: u32 = 0b000_00000_00000_11111;
    const SHIFT_X: u32 = 0;
    const MASK_Y: u32 = 0b000_00000_11111_00000;
    const SHIFT_Y: u32 = 5;
    const MASK_Z: u32 = 0b000_11111_00000_00000;
    const SHIFT_Z: u32 = 5+5;
    const MASK_NORM: u32 = 0b111_00000_00000_00000;
    const SHIFT_NORM: u32 = 5+5+5;

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
        ((x & (Self::MASK_X >> Self::SHIFT_X)) << Self::SHIFT_X) |
            ((y & (Self::MASK_Y >> Self::SHIFT_Y)) << Self::SHIFT_Y) |
            ((z & (Self::MASK_Z >> Self::SHIFT_Z)) << Self::SHIFT_Z) |
            ((dir.index() & (Self::MASK_NORM >> Self::SHIFT_NORM)) << Self::SHIFT_NORM)
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
struct ObjectIndexUBO {
    index: u32
}

#[derive(BufferContents, Clone, Copy, Default)]
#[repr(C)]
struct MaterialUBO {
    texture_index: u32
}

struct RenderInfo {
    chunk_mesh: [Option<ChunkMesh>; 6],
    chunk_pos: IVec3,
    index: u32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ChunkMesh {
    vertex_buffer_alloc: ChunkBufferAlloc,
    index_buffer_alloc: ChunkBufferAlloc,
    index_count: u32,
}

impl Default for ChunkMesh {
    fn default() -> Self {
        ChunkMesh {
            vertex_buffer_alloc: NULL_CHUNK_BUFFER_ALLOC,
            index_buffer_alloc: NULL_CHUNK_BUFFER_ALLOC,
            index_count: 0,
        }
    }
}

#[derive(Clone)]
struct BatchedDrawCommand {
    mesh: Mesh<VoxelVertex>,
    first_instance: u32,
    instance_count: u32,
}


#[derive(Component, Clone, Copy)]
pub struct VoxelChunkChangedMarker;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChunkBufferRegion {
    buffer_offset: DeviceSize,
    buffer_size: DeviceSize,
    alignment: Option<DeviceAlignment>,
}

const NULL_CHUNK_BUFFER_REGION: ChunkBufferRegion = ChunkBufferRegion{
    buffer_offset: 0,
    buffer_size: 0,
    alignment: None,
};

impl ChunkBufferRegion {
    pub fn buffer_start_aligned(&self) -> DeviceSize {
        if let Some(alignment) = self.alignment {
            align_up(self.buffer_offset, alignment)
        } else {
            self.buffer_offset
        }
    }

    pub fn buffer_end_aligned(&self) -> DeviceSize {
        if let Some(alignment) = self.alignment {
            align_down(self.buffer_offset + self.buffer_size, alignment)
        } else {
            self.buffer_offset + self.buffer_size
        }
    }

    pub fn buffer_size_aligned(&self) -> DeviceSize {
        self.buffer_end_aligned() - self.buffer_start_aligned()
    }
}

pub struct VoxelRenderer {
    solid_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    wire_graphics_pipeline: Option<Arc<GraphicsPipeline>>,
    resources: Vec<FrameResource>,

    chunk_buffer_handler: ChunkMeshBufferHandler,

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
    object_count: u32,
    max_object_count: u32,
    max_material_count: u32,
    max_draw_commands_count: u32,
    scene_changed: bool,

    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_frame_complete: Option<ReaderId<FrameCompleteEvent>>,

    null_texture: Arc<ImageView>,
    default_sampler: Arc<Sampler>,
}

struct FrameResource {
    buffer_object_data: Option<Subbuffer<[ObjectDataUBO]>>,
    buffer_object_indices: Option<Subbuffer<[ObjectIndexUBO]>>,
    buffer_material_data: Option<Subbuffer<[MaterialUBO]>>,
    buffer_indirect_draw_commands: Option<Subbuffer<[DrawIndexedIndirectCommand]>>,
    descriptor_set_camera: Option<Arc<DescriptorSet>>,
    descriptor_set_world: Option<Arc<DescriptorSet>>,
    descriptor_set_materials: Option<Arc<DescriptorSet>>,

    descriptor_writes_world: Vec<WriteDescriptorSet>,
    descriptor_writes_materials: Vec<WriteDescriptorSet>,

    recreate_descriptor_sets: bool,
    scene_changed: bool,
    textures_changed: bool,
    materials_changed: bool,
    camera_hash: u64,

    active_resources: Vec<Arc<dyn Any>>,
}

// Manage a list of global chunk data, and allocate new buffers when we cannot fit chunks into the current buffers.
struct ChunkMeshBufferHandler {
    allocator: Arc<dyn MemoryAllocator>,
    vertex_buffers: Vec<ChunkMeshBuffer>,
    index_buffers: Vec<ChunkMeshBuffer>,
    // buffers_by_free_space: BTreeMap<DeviceSize, usize>,
    batch_free_list: Vec<ChunkBufferAlloc>,

    indirect_draw_commands: HashMap<(usize, usize), ChunkMeshIndirectDrawCommands>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ChunkMeshIndirectDrawCommands {
    commands: Vec<DrawIndexedIndirectCommand>,
    indirect_buffer_offset: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ChunkMeshBufferType {
    VertexBuffer,
    IndexBuffer,
}

impl ChunkMeshBufferHandler {
    fn new(allocator: Arc<dyn MemoryAllocator>) -> Result<Self> {
        Ok(ChunkMeshBufferHandler {
            allocator,
            vertex_buffers: vec![],
            index_buffers: vec![],

            batch_free_list: vec![],

            indirect_draw_commands: HashMap::new(),
        })
    }

    pub fn indirect_draw_commands(&mut self, vertex_buffer_index: usize, index_buffer_index: usize) -> &mut ChunkMeshIndirectDrawCommands {
        assert!(vertex_buffer_index < self.vertex_buffers.len() && index_buffer_index < self.index_buffers.len());
        self.indirect_draw_commands.entry((vertex_buffer_index, index_buffer_index)).or_default()
    }

    pub fn clear_indirect_draw_commands(&mut self) {
        for (_key, commands) in self.indirect_draw_commands.iter_mut() {
            commands.commands.clear();
        }
    }

    pub fn free_batch(&mut self) {
        for mesh in self.batch_free_list.iter() {
            match mesh.buffer_type {
                ChunkMeshBufferType::VertexBuffer => self.vertex_buffers[mesh.buffer_index].free(mesh.buffer_region),
                ChunkMeshBufferType::IndexBuffer => self.index_buffers[mesh.buffer_index].free(mesh.buffer_region)
            }
        }
        self.batch_free_list.clear();
    }

    fn find_buffer_for_required_size(buffers: &Vec<ChunkMeshBuffer>, required_size: DeviceSize) -> Option<usize> {
        for (i, buffer) in buffers.iter().enumerate() {
            let free_space = buffer.get_min_free_space(required_size);
            if free_space < required_size {
                continue;
            }

            return Some(i);
        }

        None
    }

    fn get_or_allocate_buffer(allocator: Arc<dyn MemoryAllocator>, buffers: &mut Vec<ChunkMeshBuffer>, required_size: DeviceSize, buffer_type: ChunkMeshBufferType) -> Result<usize> {

        let mut buffer_index = Self::find_buffer_for_required_size(buffers, required_size);

        if buffer_index.is_none() {
            buffer_index = Some(buffers.len());

            const MB: DeviceSize = 1024 * 1024;
            let buffer = ChunkMeshBuffer::new(allocator, 128 * MB)?;
            buffers.push(buffer);
        }

        Ok(buffer_index.unwrap())
    }

    fn allocate_multi_chunk_mesh(&mut self, buffer_type: ChunkMeshBufferType, required_size: DeviceSize, alignment: Option<DeviceAlignment>, count: u32, out_alloc_meshes: &mut Vec<ChunkBufferAlloc>) -> Result<u32> {
        if count == 0 || required_size == 0 {
            return Ok(0);
        }

        let mut remaining_count = count;
        let mut alloc_regions = Vec::with_capacity(count as usize);

        let buffers = match buffer_type {
            ChunkMeshBufferType::VertexBuffer => &mut self.vertex_buffers,
            ChunkMeshBufferType::IndexBuffer => &mut self.index_buffers
        };

        while remaining_count > 0 {
            if let Ok(buffer_index) = Self::get_or_allocate_buffer(self.allocator.clone(), buffers, required_size, buffer_type) {

                let buffer = &mut buffers[buffer_index];

                alloc_regions.clear();
                remaining_count -= buffer.allocate_multi(required_size, alignment, remaining_count, &mut alloc_regions);

                for region in &alloc_regions {
                    let chunk_mesh = ChunkBufferAlloc {
                        buffer_region: region.clone(),
                        buffer_index,
                        buffer_type
                    };

                    out_alloc_meshes.push(chunk_mesh);
                }

            } else {
                // Allocating a new buffer failed, presumably we ran out of vRAM if we reach here?
                break;
            }
        }

        let num_allocated = count - remaining_count;
        Ok(num_allocated)
    }

    pub fn allocate_chunk_mesh(&mut self, buffer_type: ChunkMeshBufferType, required_size: DeviceSize, alignment: Option<DeviceAlignment>) -> Result<Option<ChunkBufferAlloc>> {

        let buffers = match buffer_type {
            ChunkMeshBufferType::VertexBuffer => &mut self.vertex_buffers,
            ChunkMeshBufferType::IndexBuffer => &mut self.index_buffers
        };

        let buffer_index = Self::get_or_allocate_buffer(self.allocator.clone(), buffers, required_size, buffer_type)?;
        let buffer = &mut buffers[buffer_index];

        if let Some(region) = buffer.allocate(required_size, alignment) {
            let chunk_mesh = ChunkBufferAlloc {
                buffer_region: region,
                buffer_index,
                buffer_type
            };

            return Ok(Some(chunk_mesh));
        }

        Ok(None)
    }

    pub fn batch_free(&mut self, buffer_alloc: ChunkBufferAlloc) {
        self.batch_free_list.push(buffer_alloc);
    }

    pub fn get_gpu_buffer_region(&self, chunk_mesh: &ChunkBufferAlloc) -> Option<Subbuffer<[u8]>> {
        let buffer = match chunk_mesh.buffer_type {
            ChunkMeshBufferType::VertexBuffer => &self.vertex_buffers[chunk_mesh.buffer_index],
            ChunkMeshBufferType::IndexBuffer => &self.index_buffers[chunk_mesh.buffer_index]
        };

        buffer.get_gpu_buffer_region(&chunk_mesh.buffer_region)
    }

    pub fn get_gpu_buffer(&self, buffer_index: usize, buffer_type: ChunkMeshBufferType) -> Subbuffer<[u8]> {
        match buffer_type {
            ChunkMeshBufferType::VertexBuffer => self.vertex_buffers[buffer_index].buffer.clone(),
            ChunkMeshBufferType::IndexBuffer => self.index_buffers[buffer_index].buffer.clone()
        }
    }

    pub fn draw_debug_gui(&self, ticker: &mut Ticker, ctx: &Context) {

        egui::Window::new("Voxel Memory Debug")
            // .anchor(egui::Align2::LEFT_BOTTOM, [10.0, 10.0])
            // .default_size([400.0, 200.0])
            .show(ctx, |ui| {

                let mut draw_buffer_region = |i: usize, buffer: &ChunkMeshBuffer, buffer_type: ChunkMeshBufferType| {

                    ui.collapsing(format!("Chunk Mesh Buffer {i} ({buffer_type:?})").as_str(), |ui| {
                        let buffer_size_bytes = buffer.buffer.size();
                        let (buffer_size, buffer_size_unit) = util::format_size_bytes(buffer_size_bytes);

                        let free_space_bytes = buffer.total_free_space();
                        let (free_space, free_space_unit) = util::format_size_bytes(free_space_bytes);
                        let free_percent = free_space_bytes as f64 / buffer_size_bytes as f64 * 100.0;

                        let used_space_bytes = buffer_size_bytes - free_space_bytes;
                        let (used_space, used_space_unit) = util::format_size_bytes(used_space_bytes);
                        let used_percent = used_space_bytes as f64 / buffer_size_bytes as f64 * 100.0;

                        let max_space_bytes = buffer.get_max_free_space();
                        let (max_space, max_space_unit) = util::format_size_bytes(max_space_bytes);

                        let min_space_bytes = buffer.get_min_free_space(0);
                        let (min_space, min_space_unit) = util::format_size_bytes(min_space_bytes);

                        let free_region_fragmentation = buffer.calculate_fragmentation_estimate(true) * 100.0;
                        // let total_fragmentation = buffer.calculate_fragmentation_estimate(false) * 100.0;

                        let num_used_regions = buffer.used_regions.len();
                        let num_free_regions = buffer.free_regions_by_offset.len();

                        let mut smallest_used_region = DeviceSize::MAX;
                        let mut largest_used_region = DeviceSize::MIN;
                        let avg_used_region_size = (used_space_bytes as f64 / num_used_regions as f64) as DeviceSize;

                        for (_offset_key, region) in buffer.used_regions.iter() {
                            smallest_used_region = DeviceSize::min(smallest_used_region, region.buffer_size);
                            largest_used_region = DeviceSize::max(largest_used_region, region.buffer_size);
                        }

                        let mut smallest_free_region = DeviceSize::MAX;
                        let mut largest_free_region = DeviceSize::MIN;
                        let avg_free_region_size = (free_space_bytes as f64 / num_free_regions as f64) as DeviceSize;

                        for (_offset_key, region) in buffer.free_regions_by_offset.iter() {
                            smallest_free_region = DeviceSize::min(smallest_free_region, region.buffer_size);
                            largest_free_region = DeviceSize::max(largest_free_region, region.buffer_size);
                        }

                        ui.horizontal(|ui| {

                            ui.vertical(|ui| {
                                ui.label(format!("Buffer capacity: {buffer_size_bytes} ({buffer_size:.2} {buffer_size_unit})"));
                                ui.label(format!("Total used space: {used_space_bytes} ({used_space:.2} {used_space_unit} / {used_percent:.2}%)"));
                                ui.label(format!("Total free space: {free_space_bytes} ({free_space:.2} {free_space_unit} / {free_percent:.2}%)"));
                                ui.label(format!("Max free region: {max_space_bytes} ({max_space:.2} {max_space_unit})"));
                                ui.label(format!("Min free region: {min_space_bytes} ({min_space:.2} {min_space_unit})"));
                                ui.label(format!("Used regions: {num_used_regions} - Min {smallest_used_region} bytes, Max {largest_used_region} bytes, Avg {avg_used_region_size} bytes"));
                                ui.label(format!("Free regions: {num_free_regions} - Min {smallest_free_region} bytes, Max {largest_free_region} bytes, Avg {avg_free_region_size} bytes"));
                                ui.label(format!("Fragmentation: {free_region_fragmentation:.3}%"));
                            });

                            let (rect, _) = ui.allocate_exact_size(egui::Vec2::new(600.0, 300.0), egui::Sense::hover());
                            let painter = ui.painter_at(rect);

                            let cell_width = 1;
                            let cell_height = 2;
                            let cell_count_x = (rect.width() as DeviceSize) / cell_width;
                            let cell_count_y = (rect.height() as DeviceSize) / cell_height;
                            let cell_count = cell_count_x * cell_count_y;
                            let cell_size_bytes = f64::ceil((buffer.buffer.size() / cell_count) as f64) as DeviceSize;

                            let draw_span = |start_x: DeviceSize, end_x: DeviceSize, row: DeviceSize, colour: egui::Color32| {
                                let offset = painter.clip_rect().min;
                                let p0 = egui::Pos2::new(offset.x + (start_x * cell_width) as f32, offset.y + (row * cell_height) as f32);
                                let p1 = egui::Pos2::new(offset.x + (end_x * cell_width) as f32, offset.y + ((row + 1) * cell_height) as f32);
                                let r = egui::Rect::from_min_max(p0, p1);
                                painter.rect_filled(r, 0.0, colour);
                                // painter.line_segment([egui::Pos2::new(r.right(), p0.y), egui::Pos2::new(r.right(), p1.y)], egui::Stroke::new(1.0, colour.linear_multiply(0.5)));
                                // painter.line_segment([egui::Pos2::new(p0.x, r.bottom()), egui::Pos2::new(p1.x, r.bottom())], egui::Stroke::new(1.0, egui::Color32::DARK_GRAY));
                            };


                            let mut get_span_colour = |start_cell_index: DeviceSize, end_cell_index: DeviceSize, region: &ChunkBufferRegion| {
                                let mut hasher = DefaultHasher::new();
                                start_cell_index.hash(&mut hasher);
                                end_cell_index.hash(&mut hasher);
                                let t = hasher.finish();
                                let r = (t & 0xFF) as u8;
                                let g = ((t >> 8) & 0xFF) as u8;
                                let b = ((t >> 16) & 0xFF) as u8;
                                egui::Color32::from_rgb(r / 2 + 127, g, b / 2 + 127)

                                // let span_size_bytes = (end_cell_index - start_cell_index) * cell_size_bytes;
                                // let percent_filled = region.buffer_size as f64 / span_size_bytes as f64;
                                // let percent_filled = percent_filled * percent_filled * percent_filled;
                                //
                                // egui::Color32::GREEN.lerp_to_gamma(egui::Color32::RED, percent_filled as f32)
                            };

                            for (_offset_key, region) in buffer.used_regions.iter() {
                                let start_cell_index = region.buffer_offset / cell_size_bytes;
                                let end_cell_index = (region.buffer_offset + region.buffer_size) / cell_size_bytes;

                                let start_cell_x = start_cell_index % cell_count_x;
                                let start_cell_y = start_cell_index / cell_count_x;
                                let end_cell_x = end_cell_index % cell_count_x;
                                let end_cell_y = end_cell_index / cell_count_x;

                                let colour = get_span_colour(start_cell_index, end_cell_index, region);
                                if start_cell_y == end_cell_y {
                                    // Span is on one row
                                    draw_span(start_cell_x, end_cell_x, start_cell_y, colour);

                                } else {
                                    // We span across multiple rows, the start and end need to be handled

                                    let mut curr_cell_y = start_cell_y;

                                    // Draw the first span from the start_x to the right edge
                                    draw_span(start_cell_x, cell_count_x, curr_cell_y, colour);
                                    curr_cell_y += 1;


                                    while curr_cell_y != end_cell_y {
                                        // Draw in-between spans from the left edge to the right edge
                                        draw_span(0, cell_count_x, curr_cell_y, colour);
                                        curr_cell_y += 1;
                                    }

                                    // Draw the last span from the left edge to end_x
                                    draw_span(0, end_cell_x, curr_cell_y, colour);

                                }
                            }
                        });

                    });
                };
                for (i, buffer) in self.vertex_buffers.iter().enumerate() {
                    draw_buffer_region(i, buffer, ChunkMeshBufferType::VertexBuffer);
                }
                for (i, buffer) in self.index_buffers.iter().enumerate() {
                    draw_buffer_region(i, buffer, ChunkMeshBufferType::IndexBuffer);
                }
            });
    }
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ChunkBufferAlloc {
    buffer_region: ChunkBufferRegion,
    buffer_index: usize,
    buffer_type: ChunkMeshBufferType,
    // vertex_buffer: Subbuffer<[u8]>,
    // index_buffer: Subbuffer<[u8]>,
}

const NULL_CHUNK_BUFFER_ALLOC: ChunkBufferAlloc = ChunkBufferAlloc {
    buffer_region: NULL_CHUNK_BUFFER_REGION,
    buffer_index: 0,
    buffer_type: ChunkMeshBufferType::VertexBuffer,
};



// A single large GPU buffer to store the mesh data for many chunks
struct ChunkMeshBuffer {
    buffer: Subbuffer<[u8]>,
    used_regions: HashMap<DeviceSize, ChunkBufferRegion>,
    free_regions_by_size: BTreeMap<DeviceSize, Vec<ChunkBufferRegion>>,
    free_regions_by_offset: BTreeMap<DeviceSize, ChunkBufferRegion>,

    total_free_space: DeviceSize, // The total amount of free space in the buffer, including fragmentation
}

impl ChunkMeshBuffer {
    fn new(allocator: Arc<dyn MemoryAllocator>, size_bytes: DeviceSize) -> Result<Self> {
        let create_info = BufferCreateInfo{
            // flags: TODO investigate SPARSE_BINDING
            usage: BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        };

        let buffer = Buffer::new_slice::<u8>(allocator, create_info, allocation_info, size_bytes)?;

        // First free region is the whole buffer
        let free_region = ChunkBufferRegion {
            buffer_offset: 0,
            buffer_size: buffer.size(),
            alignment: DeviceAlignment::new(0),
        };

        let chunk_mesh_buffer = ChunkMeshBuffer {
            buffer,
            used_regions: HashMap::new(),
            free_regions_by_size: BTreeMap::from([( free_region.buffer_size, vec![ free_region ] )]),
            free_regions_by_offset: BTreeMap::from([(free_region.buffer_offset, free_region)]),

            total_free_space: free_region.buffer_size,
        };

        Ok(chunk_mesh_buffer)
    }

    fn get_gpu_buffer_region(&self, buffer_region: &ChunkBufferRegion) -> Option<Subbuffer<[u8]>> {
        let buf = self.buffer.clone();
        let range_start = buffer_region.buffer_offset;
        let range_end = range_start + buffer_region.buffer_size;

        let buf = buf.slice( range_start .. range_end);
        Some(buf)
    }

    fn sort_regions_by_size_asc(a: &ChunkBufferRegion, b: &ChunkBufferRegion) -> Ordering {
        a.buffer_size.cmp(&b.buffer_size)
    }
    fn sort_regions_by_size_desc(a: &ChunkBufferRegion, b: &ChunkBufferRegion) -> Ordering {
        b.buffer_size.cmp(&a.buffer_size)
    }
    fn sort_regions_by_offset_asc(a: &ChunkBufferRegion, b: &ChunkBufferRegion) -> Ordering {
        a.buffer_offset.cmp(&b.buffer_offset)
    }
    fn sort_regions_by_offset_desc(a: &ChunkBufferRegion, b: &ChunkBufferRegion) -> Ordering {
        b.buffer_offset.cmp(&a.buffer_offset)
    }

    /// Find the smallest free regions larger than the required size. There may be multiple regions of the same size, all of
    /// which will be returned in a vector. The vector is expected to be sorted by offset in descending order, since all
    /// regions are the same size. The last element in the returned vector will be the lowest offset into the buffer
    /// Returns (size_key, Vec<free_region>) if a valid region was found
    /// Returns None if no regions were available and large enough
    fn find_free_regions(free_regions_by_size: &mut BTreeMap<DeviceSize, Vec<ChunkBufferRegion>>, required_size: DeviceSize) -> Option<(DeviceSize, &mut Vec<ChunkBufferRegion>)>{

        for (size, regions) in free_regions_by_size.range_mut(required_size..) {
            if regions.is_empty() {
                // This shouldn't happen if the regions are managed correctly.
                continue;
            }

            let region = regions.last().unwrap();
            debug_assert!(region.buffer_size == *size && region.buffer_size >= required_size);

            return Some((*size, regions))
        }

        None
    }

    fn get_max_free_space(&self) -> DeviceSize {
        if let Some((size_key, _)) = self.free_regions_by_size.last_key_value() {
            *size_key
        } else {
            0
        }
    }

    fn get_min_free_space(&self, required_size: DeviceSize) -> DeviceSize {
        for (size, regions) in self.free_regions_by_size.range(required_size..) {
            if !regions.is_empty() {
                return *size;
            }
        }

        0
    }

    fn total_free_space(&self) -> DeviceSize {
        self.total_free_space
    }

    fn calculate_fragmentation_estimate(&self, only_check_free_regions: bool) -> f64 {
        // Implementation from here: https://asawicki.info/news_1757_a_metric_for_memory_fragmentation

        let mut quality = 0;
        let mut total_free_size = 0;

        for (offset_key, region) in self.free_regions_by_offset.iter() {
            quality += region.buffer_size * region.buffer_size;
            total_free_size += region.buffer_size;
        }

        if !only_check_free_regions {
            total_free_size = self.buffer.size();
        }

        if total_free_size == 0 {
            // Avoid divide-by-zero. All space is used, this is 0% fragmentation.
            return 0.0;
        }

        let quality = f64::sqrt(quality as f64) / total_free_size as f64;
        let fragmentation = 1.0 - (quality * quality);

        fragmentation
    }

    fn align_up_opt(val: DeviceSize, alignment: Option<DeviceAlignment>) -> DeviceSize {
        if let Some(alignment) = alignment {
            align_up(val, alignment)
        } else {
            val
        }
    }

    fn align_down_opt(val: DeviceSize, alignment: Option<DeviceAlignment>) -> DeviceSize {
        if let Some(alignment) = alignment {
            align_down(val, alignment)
        } else {
            val
        }
    }

    fn check_contiguous_regions(regions: &[&ChunkBufferRegion]) -> bool {
        if regions.len() > 1 {
            for i in 1..regions.len() {
                let prev_region = regions[i - 1];
                let curr_region = regions[i];

                let prev_region_end = prev_region.buffer_offset + prev_region.buffer_size;
                let curr_region_begin = curr_region.buffer_offset;

                if prev_region_end != curr_region_begin {
                    return false;
                }
            }
        }

        true
    }

    fn merge_contiguous_neighbours(
        free_regions_by_size: &mut BTreeMap<DeviceSize, Vec<ChunkBufferRegion>>,
        free_regions_by_offset: &mut BTreeMap<DeviceSize, ChunkBufferRegion>,
        offset: DeviceSize
    ) -> Option<ChunkBufferRegion> {

        // debug!("merge_contiguous_neighbours offset={offset}");
        // debug!("free_regions_by_size: {free_regions_by_size:?}");
        // debug!("free_regions_by_offset: {free_regions_by_offset:?}");

        let mut it = free_regions_by_offset.range_mut(..=offset);

        // Get the first region, this should include the supplied offset
        let curr_region = it.next_back();
        if curr_region.is_none() {
            // There was no starting region. The supplied offset was less than any entries in the free_regions_by_offset map
            return None;
        }

        let mut curr_region = curr_region.unwrap().1;

        // Iterate backwards through the adjacent regions to find the start of this contiguous section
        while let Some((_offset_key, prev_region)) = it.next_back() {
            if !Self::check_contiguous_regions(&[&*prev_region, &*curr_region]) {
                break;
            }
            curr_region = prev_region;
        }

        let start_offset = curr_region.buffer_offset;

        // We now iterate forwards through all contiguous regions and merge them as we go...
        let mut it = free_regions_by_offset.range_mut(start_offset..);
        let mut curr_region = it.next().unwrap().1;

        // List to track the region size keys that need to be fixed after.
        let mut buffer_sizes_to_clean = HashSet::new();

        let original_buffer_size = curr_region.buffer_size;

        let mut cleanup_free_regions_by_size = |removed_region: &ChunkBufferRegion| {
            // Zero out the size of this region in the free_regions_by_size map, which marks it to be removed after
            free_regions_by_size.entry(removed_region.buffer_size).and_modify(|regions| {
                buffer_sizes_to_clean.insert(removed_region.buffer_size);

                let index = regions.binary_search_by(|region| {
                    Self::sort_regions_by_offset_desc(region, removed_region)
                    // region.buffer_offset.cmp(&next_region.buffer_offset)
                });
                let index = index.unwrap(); // This element must be found, otherwise we have lost track somewhere

                regions[index].buffer_size = 0;
            });
        };


        while let Some((_offset_key, next_region)) = it.next() {
            if !Self::check_contiguous_regions(&[&*curr_region, &*next_region]) {
                // Reached the end.
                break;
            }

            if curr_region.buffer_size == original_buffer_size {
                // Be sure to clean up the starting region now that we are modifying it.
                cleanup_free_regions_by_size(curr_region);
            }

            // Grow the current region
            curr_region.buffer_size += next_region.buffer_size;

            cleanup_free_regions_by_size(next_region);

            // Zero out the size of this region in the free_regions_by_offset map, which marks it to be removed after
            next_region.buffer_size = 0;
        }

        let curr_region = curr_region.clone();

        // Filter out zero-size regions from the free_regions_by_size map
        for size_key in &buffer_sizes_to_clean {
            let mut do_remove = false;
            free_regions_by_size.entry(*size_key).and_modify(|regions| {
                regions.retain(|region| region.buffer_size != 0);
                if regions.is_empty() {
                    do_remove = true;
                }
            });
            if do_remove {
                free_regions_by_size.remove(size_key);
            }
        }

        free_regions_by_offset.retain(|_offset_key, region| {
            region.buffer_size != 0
        });

        if curr_region.buffer_size != original_buffer_size {
            let regions = free_regions_by_size.entry(curr_region.buffer_size).or_default();
            // regions.insert(regions.partition_point(Self::sort_regions_by_offset_desc), curr_region);
            regions.push(curr_region);
            regions.sort_by(Self::sort_regions_by_offset_desc);
        }


        Some(curr_region)
    }

    /// Allocate a sub-region of this buffer with required_size bytes.
    /// Returns the allocated region, or None if no allocation could be made
    fn allocate(&mut self, required_size: DeviceSize, alignment: Option<DeviceAlignment>) -> Option<ChunkBufferRegion> {

        // self.debug_sanity_check();

        // Find the smallest region that is big enough for the required allocation size
        let curr_regions = Self::find_free_regions(&mut self.free_regions_by_size, required_size);
        if curr_regions.is_none() {
            // No region was found, unable to allocate
            return None;
        }

        let (size_key, free_regions) = curr_regions.unwrap();

        let mut free_region = free_regions.pop().unwrap();
        if free_regions.is_empty() {
            self.free_regions_by_size.remove(&size_key);
        }

        {
            let r = self.free_regions_by_offset.remove(&free_region.buffer_offset).unwrap();
            debug_assert_eq!(r.buffer_size, free_region.buffer_size); // Sanity check, the same region must have been free in both maps
        }

        let aligned_range_begin = Self::align_up_opt(free_region.buffer_offset, alignment);
        let aligned_range_end = Self::align_up_opt(aligned_range_begin + required_size, alignment);

        let alloc_region = ChunkBufferRegion {
            buffer_offset: free_region.buffer_offset,
            buffer_size: aligned_range_end - aligned_range_begin,
            alignment
        };

        // Shrink the free region we allocated from
        free_region.buffer_offset += alloc_region.buffer_size;
        free_region.buffer_size -= alloc_region.buffer_size;

        // Track this allocated region as used
        self.used_regions.insert(alloc_region.buffer_offset, alloc_region.clone());

        if free_region.buffer_size > 0 {
            // If the free region has remaining space, we make sure to keep track of it.
            // Re-insert to both the free_by_size and free_by_offset maps.
            let free_regions = self.free_regions_by_size.entry(free_region.buffer_size).or_default();
            free_regions.push(free_region);
            free_regions.sort_by(Self::sort_regions_by_offset_desc); // Ensure all free regions for this size are sorted by offset descending

            let r = self.free_regions_by_offset.insert(free_region.buffer_offset, free_region);
            debug_assert_eq!(r, None); // Sanity check, there must not have been a previous entry here, otherwise we have lost track of something.
        }

        self.total_free_space -= alloc_region.buffer_size;

        // self.debug_sanity_check();

        Some(alloc_region)
    }

    /// Allocate multiple (count) sub-regions of this buffer with required_size bytes.
    /// The successfully allocated regions are pushed into the provided out_alloc_regions vector
    /// Returns the number of regions that were successfully allocated. This may be less than the requested count if this buffer runs out of space.
    fn allocate_multi(&mut self, required_size: DeviceSize, alignment: Option<DeviceAlignment>, count: u32, out_alloc_regions: &mut Vec<ChunkBufferRegion>) -> u32 {

        // self.debug_sanity_check();

        let mut curr_regions: Option<(DeviceSize, &mut Vec<ChunkBufferRegion>)> = None;

        let mut updated_free_regions: HashMap<DeviceSize, Vec<(DeviceSize, DeviceSize, ChunkBufferRegion)>> = HashMap::new();

        let mut curr_region_start_offset: Option<DeviceSize> = None;

        let minimum_size = Self::align_up_opt(required_size, alignment);

        let mut num_allocated = 0;
        for i in 0..count {

            if curr_regions.is_none() {
                // Find the smallest region that is big enough for the required allocation size
                curr_regions = Self::find_free_regions(&mut self.free_regions_by_size, minimum_size);
                if curr_regions.is_none() {
                    // No regions big enough were found, we cannot allocate any more regions.
                    break;
                }
            }

            let (size_key, free_regions) = curr_regions.as_mut().unwrap();

            // Retrieve the last entry in the free_regions list for this size_key.
            // This is expected to be the lowest offset in the buffer
            let free_region = free_regions.last_mut().unwrap();
            debug_assert!(free_region.buffer_size >= minimum_size);

            if curr_region_start_offset.is_none() {
                // Keep track of the starting offset so we can later fix the free_regions_by_offset map
                curr_region_start_offset = Some(free_region.buffer_offset);
            }

            let aligned_range_begin = Self::align_up_opt(free_region.buffer_offset, alignment);
            let aligned_range_end = Self::align_up_opt(aligned_range_begin + required_size, alignment);

            let alloc_region = ChunkBufferRegion {
                buffer_offset: free_region.buffer_offset,
                buffer_size: aligned_range_end - aligned_range_begin,
                alignment
            };

            // Shrink the free region we allocated from
            free_region.buffer_offset += alloc_region.buffer_size;
            free_region.buffer_size -= alloc_region.buffer_size;

            // Track this allocated region as used
            self.used_regions.insert(alloc_region.buffer_offset, alloc_region.clone());
            self.total_free_space -= alloc_region.buffer_size;
            out_alloc_regions.push(alloc_region);
            num_allocated += 1;

            let aligned_range_begin = Self::align_up_opt(free_region.buffer_offset, alignment);
            let aligned_range_end = Self::align_down_opt(free_region.buffer_offset + free_region.buffer_size, alignment);
            let remaining_region_size = aligned_range_end - aligned_range_begin;
            if remaining_region_size < minimum_size || i == count - 1 {
                // Either the current free region is too small to use, or we reached the final iteration.
                // In either case, we need to be sure to update the tracked free regions.

                let new_size_key = free_region.buffer_size;
                let original_size_key = *size_key;
                let original_offset_key = curr_region_start_offset.take().unwrap();
                let new_free_region = free_regions.pop().unwrap(); // Pop it from the list.

                if free_regions.is_empty() {
                    // Reset current_region to None so a new region is selected on the next loop, else, we know
                    // all regions in the list are the same size, so we will use the next one in the next loop
                    curr_regions = None;
                }

                // Add the size_key and free region to the updated_free_regions. We will
                // later use this to fix up the actual free regions lists of this struct
                updated_free_regions.entry(new_size_key).or_default().push((original_size_key, original_offset_key, new_free_region));
            }
        }

        for (size_key, updates) in updated_free_regions {

            // Erase the free updated regions from the free_regions_by_offset map
            for (original_size_key, original_offset_key, _) in &updates {
                let r = self.free_regions_by_offset.remove(&original_offset_key).unwrap();
                debug_assert_eq!(r.buffer_size, *original_size_key); // Sanity check, the same region must have been free in both maps
            }

            if size_key > 0 {
                // Insert the new free regions after the allocations.
                let free_regions = self.free_regions_by_size.entry(size_key).or_default();
                for (_, _, new_free_region) in &updates {
                    free_regions.push(*new_free_region);

                    let r = self.free_regions_by_offset.insert(new_free_region.buffer_offset, *new_free_region);
                    debug_assert_eq!(r, None); // Sanity check, there must not have been a previous entry here, otherwise we have lost track of something.
                }
                free_regions.sort_by(Self::sort_regions_by_offset_desc); // Ensure all free regions for this size are sorted by offset descending
            }

            // Erase the free regions lists that became empty after the allocations
            for (original_size_key, _, _) in &updates {
                if let Some(s) = self.free_regions_by_size.get(original_size_key) {
                    if s.is_empty() {
                        self.free_regions_by_size.remove(original_size_key);
                    }
                }
            }
        }

        // self.debug_sanity_check();
        num_allocated
    }

    fn free(&mut self, alloc_region: ChunkBufferRegion) {
        // self.debug_sanity_check();

        if let Some(used_region) = self.used_regions.remove(&alloc_region.buffer_offset) {
            debug_assert_eq!(used_region, alloc_region);

            let r = self.free_regions_by_offset.insert(used_region.buffer_offset, used_region);
            debug_assert_eq!(r, None); // Sanity check, this region should not have already been in the map.

            let free_regions = self.free_regions_by_size.entry(used_region.buffer_size).or_default();
            free_regions.push(used_region);
            free_regions.sort_by(Self::sort_regions_by_offset_desc);
            self.total_free_space += alloc_region.buffer_size;

            Self::merge_contiguous_neighbours(&mut self.free_regions_by_size, &mut self.free_regions_by_offset, used_region.buffer_offset);
            // self.debug_sanity_check();
        }
    }

    fn debug_sanity_check(&self) {
        let mut count_by_offset = 0;

        for (_offset_key, region_by_offset) in &self.free_regions_by_offset {
            let size_key = region_by_offset.buffer_size;
            let r = self.free_regions_by_size.get(&size_key);
            assert!(r.is_some(), "free_regions_by_offset entry ({region_by_offset:?}) has no corresponding size key in free_regions_by_size:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);
            let regions_by_size = r.unwrap();

            count_by_offset += 1;

            let is_sorted = regions_by_size.is_sorted_by(|a, b| Self::sort_regions_by_offset_desc(a, b).is_le());
            assert!(is_sorted, "Regions for size_key {} are not sorted:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", size_key, self.free_regions_by_offset, self.free_regions_by_size);

            let mut dedup_regions_by_size = regions_by_size.clone();
            dedup_regions_by_size.dedup_by(|a, b| a.buffer_offset == b.buffer_offset);

            assert_eq!(dedup_regions_by_size.len(), regions_by_size.len(), "Regions for size_key {size_key} contains duplicates:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);

            let r = regions_by_size.binary_search_by(|region_by_size| {
                Self::sort_regions_by_offset_desc(region_by_size, region_by_offset)
            });

            assert!(r.is_ok(), "free_regions_by_offset entry ({region_by_offset:?}) has no corresponding entry in free_regions_by_size list for size_key {size_key}:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);

            let region_by_size = regions_by_size[r.unwrap()];
            assert_eq!(region_by_size, *region_by_offset, "The same entry in free_regions_by_offset ({region_by_offset:?}) and free_regions_by_size ({region_by_size:?}) does not match\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);
        }

        let mut count_by_size = 0;
        for (_offset_key, regions_by_size) in &self.free_regions_by_size {
            for region_by_size in regions_by_size {
                let region_by_offset = self.free_regions_by_offset.get(&region_by_size.buffer_offset);
                assert!(region_by_offset.is_some(), "free_regions_by_size entry ({regions_by_size:?}) has no corresponding offset key in free_regions_by_offset:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);

                let region_by_offset = region_by_offset.unwrap();
                assert_eq!(*region_by_size, *region_by_offset, "The same entry in free_regions_by_offset ({region_by_offset:?}) and free_regions_by_size ({region_by_size:?}) does not match\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);
            }
            count_by_size += regions_by_size.len();
        }

        assert_eq!(count_by_offset, count_by_size, "free_regions_by_offset and free_regions_by_size maps are out of sync:\nfree_regions_by_offset={:?}\nfree_regions_by_size={:?}\n", self.free_regions_by_offset, self.free_regions_by_size);
    }
}




impl VoxelRenderer {
    pub fn new(graphics: &GraphicsManager) -> Result<Self> {

        let null_texture = Self::create_null_texture(graphics)?;
        let default_sampler = Self::create_default_sampler(graphics.device().clone())?;

        let chunk_buffer_handler = ChunkMeshBufferHandler::new(graphics.memory_allocator())?;

        let scene_renderer = VoxelRenderer{

            solid_graphics_pipeline: None,
            wire_graphics_pipeline: None,
            resources: vec![],
            // render_camera: RenderCamera::new(graphics),
            // camera: Camera::new(),

            chunk_buffer_handler,

            textures: vec![],
            materials: vec![],
            textures_map: Default::default(),
            materials_map: Default::default(),
            textures_changed: false,
            materials_changed: false,
            // indirect_draw_commands: vec![],
            render_info: vec![],
            object_data: vec![],
            object_indices: vec![],
            // meshes: vec![],

            material_data: vec![],
            object_count: 0,
            max_object_count: 100,
            max_material_count: 100,
            max_draw_commands_count: 100,
            scene_changed: false,

            event_recreate_swapchain: None,
            event_frame_complete: None,

            null_texture,
            default_sampler,

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
                buffer_indirect_draw_commands: None,
                descriptor_set_camera: None,
                descriptor_set_world: None,
                descriptor_set_materials: None,

                descriptor_writes_world: vec![],
                descriptor_writes_materials: vec![],
                
                recreate_descriptor_sets: true,
                scene_changed: false,
                textures_changed: false,
                materials_changed: false,
                camera_hash: 0,

                active_resources: vec![]
            }
        });

        Ok(())
    }

    pub fn unload_chunk_mesh(&mut self, chunk_mesh: ChunkMesh) {
        self.chunk_buffer_handler.batch_free(chunk_mesh.vertex_buffer_alloc);
        self.chunk_buffer_handler.batch_free(chunk_mesh.index_buffer_alloc);
    }

    pub fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()> {
        profile_scope_fn!(&engine.frame_profiler);

        if ticker.time_since_last_dbg() >= ticker.debug_interval() {

            let len = engine.scene.world.loaded_chunks_count();
            debug!("{len} chunks to render");
        }

        self.scene_changed = false;

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
        }


        self.chunk_buffer_handler.free_batch();

        self.allocate_chunk_buffer_regions(&mut engine.scene)?;

        self.prepare_static_scene(cmd_buf, &mut engine.scene, &engine.graphics)?;

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

        let view_pos = camera.camera.position();

        self.build_draw_commands(cmd_buf, &engine.graphics, view_pos, frame_index)?;

        if engine.wireframe_mode().is_solid() {
            let solid_graphics_pipeline = self.solid_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(solid_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene, view_pos, engine.graphics.current_frame_index())?;
        }
        if engine.wireframe_mode().is_wire() {
            let wire_graphics_pipeline = self.wire_graphics_pipeline.as_ref().unwrap();
            cmd_buf.bind_pipeline_graphics(wire_graphics_pipeline.clone())?;
            self.draw_scene(cmd_buf, &mut engine.scene, view_pos, engine.graphics.current_frame_index())?;
        }
        Ok(())
    }


    fn check_changed_camera(camera: &RenderCamera, resource: &FrameResource, frame_index: usize) -> bool {
        camera.gpu_resource_hash(frame_index) != resource.camera_hash
    }

    fn allocate_chunk_buffer_regions(&mut self, scene: &mut Scene) -> Result<()> {

        let mut scene_changed = false;

        let mut vertex_buffer_alloc_count = 0;
        let mut index_buffer_alloc_count = 0;
        let mut required_vertex_buffer_size: DeviceSize = 0;
        let mut required_index_buffer_size: DeviceSize = 0;

        // Loop over updated chunks, figure out how much space needs to be allocated in the mesh buffer, and free necessary meshes.
        for (_chunk_pos, chunk) in scene.world.loaded_chunks_mut() {
            let chunk_data = chunk.chunk_data_mut();

            for i in 0..6 {
                if !chunk_data.update_mesh(i) {
                    continue;
                }

                if let Some(mesh_data) = chunk_data.updated_mesh_data(i) {

                    scene_changed = true;

                    let vertex_buffer_size = mesh_data.get_required_vertex_buffer_size();
                    let index_buffer_size = mesh_data.get_required_index_buffer_size();

                    required_vertex_buffer_size = DeviceSize::max(required_vertex_buffer_size, vertex_buffer_size);
                    required_index_buffer_size = DeviceSize::max(required_index_buffer_size, index_buffer_size);

                    if let Some(chunk_mesh) = chunk_data.chunk_mesh_mut(i) {
                        if chunk_mesh.index_buffer_alloc.buffer_region.buffer_size < vertex_buffer_size {
                            // The existing allocated buffer region is too small. A new one will be allocated
                            self.chunk_buffer_handler.batch_free(chunk_mesh.vertex_buffer_alloc);
                            chunk_mesh.vertex_buffer_alloc = NULL_CHUNK_BUFFER_ALLOC;
                            vertex_buffer_alloc_count += 1;
                        }

                        if chunk_mesh.index_buffer_alloc.buffer_region.buffer_size >= index_buffer_size {
                            // The existing allocated buffer region is too small. A new one will be allocated
                            self.chunk_buffer_handler.batch_free(chunk_mesh.index_buffer_alloc);
                            chunk_mesh.index_buffer_alloc = NULL_CHUNK_BUFFER_ALLOC;
                            index_buffer_alloc_count += 1;
                        }
                    } else {
                        // No mesh allocated, we need both a vertex and index buffer.
                        vertex_buffer_alloc_count += 1;
                        index_buffer_alloc_count += 1;
                    }
                }
            }
        }

        required_vertex_buffer_size += required_vertex_buffer_size / 8; // +12.5% extra space
        required_index_buffer_size += required_index_buffer_size / 8; // +12.5% extra space

        // let required_size = 262144;

        self.chunk_buffer_handler.free_batch();

        let alignment = Some(VoxelVertex::LAYOUT.alignment());

        // Allocate the space in the mesh buffer needed for the newly added chunks
        let mut chunk_vertex_buffers = Vec::with_capacity(vertex_buffer_alloc_count as usize);
        let mut chunk_index_buffers = Vec::with_capacity(index_buffer_alloc_count as usize);
        if vertex_buffer_alloc_count > 0 {
            self.chunk_buffer_handler.allocate_multi_chunk_mesh(ChunkMeshBufferType::VertexBuffer, required_vertex_buffer_size, alignment, vertex_buffer_alloc_count, &mut chunk_vertex_buffers)?;
        }
        if index_buffer_alloc_count > 0 {
            self.chunk_buffer_handler.allocate_multi_chunk_mesh(ChunkMeshBufferType::IndexBuffer, required_index_buffer_size, alignment, index_buffer_alloc_count, &mut chunk_index_buffers)?;
        }

        // Assign the allocated buffer regions to the chunk render components

        for (_chunk_pos, chunk) in scene.world.loaded_chunks_mut() {
            let chunk_data = chunk.chunk_data_mut();

            for i in 0..6 {
                if chunk_data.update_mesh(i) {
                    if let Some(mesh_data) = chunk_data.updated_mesh_data(i) {
                        let required_vertex_buffer_size = mesh_data.get_required_vertex_buffer_size();
                        let required_index_buffer_size = mesh_data.get_required_index_buffer_size();

                        let chunk_mesh = chunk_data.chunk_mesh_mut(i);
                        if chunk_mesh.is_none() {
                            chunk_data.set_chunk_mesh(i, Some(ChunkMesh::default()));
                        }

                        let chunk_mesh = chunk_data.chunk_mesh_mut(i).unwrap();
                        chunk_mesh.index_count = 0;

                        if chunk_mesh.vertex_buffer_alloc.buffer_region.buffer_size < required_vertex_buffer_size {
                            chunk_mesh.vertex_buffer_alloc = chunk_vertex_buffers.pop().unwrap();
                        }

                        if chunk_mesh.index_buffer_alloc.buffer_region.buffer_size < required_index_buffer_size {
                            chunk_mesh.index_buffer_alloc = chunk_index_buffers.pop().unwrap();
                        }
                    }
                }
            }
        };

        if !scene_changed {
            scene_changed = scene.world.has_chunks_unloaded();
        }

        if scene_changed {
            self.scene_changed = true;
            for resource in &mut self.resources {
                resource.scene_changed = true;
            }
        }

        Ok(())
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

    fn prepare_static_scene(&mut self, cmd_buf: &mut CommandBuffer, scene: &mut Scene, graphics: &GraphicsManager) -> Result<()> {
        if !self.scene_changed {
            return Ok(()); // Do nothing.
        }

        self.render_info.clear();
        self.object_data.clear();
        self.object_indices.clear();

        let mut index: u32 = 0;

        // TODO: par_iter
        scene.world.loaded_chunks_mut().for_each(|(chunk_pos, chunk)| {
            let chunk_data = chunk.chunk_data_mut();

            if !chunk_data.has_mesh() {
                return; // continue
            }

            let transform = chunk.world_transform();
            _ = self.prepare_scene_object(index, cmd_buf, chunk.chunk_data_mut(), transform);
            index += 1;
        });

        // self.object_indices[start_index..].par_sort_unstable_by(|lhs_idx, rhs_idx| {
        //     let lhs = &self.render_info[lhs_idx.index as usize];
        //     let rhs = &self.render_info[rhs_idx.index as usize];
        //
        //     lhs.mesh.cmp(&rhs.mesh)
        // });

        self.object_count = index;

        Ok(())
    }

    fn upload_chunk_mesh_data(&mut self, mesh_data: &MeshData<VoxelVertex>, chunk_mesh: &mut ChunkMesh) -> Result<()> {
        if let Some(buf) = self.chunk_buffer_handler.get_gpu_buffer_region(&chunk_mesh.vertex_buffer_alloc) {
            mesh_data.upload_vertex_data(&buf)?;
        }
        if let Some(buf) = self.chunk_buffer_handler.get_gpu_buffer_region(&chunk_mesh.index_buffer_alloc) {
            mesh_data.upload_index_data(&buf)?;
        }

        chunk_mesh.index_count = mesh_data.index_count();

        Ok(())
    }

    fn prepare_scene_object(&mut self, index: u32, cmd_buf: &mut CommandBuffer, chunk_data: &mut VoxelChunkData, transform: Transform) -> Result<()> {
        let mut object_data = ObjectDataUBO::default();
        Self::update_object_date_transform(&transform, &mut object_data);

        if let Some(material) = &chunk_data.material {
            self.register_material(material);
        }

        let mut render_info = RenderInfo{
            chunk_mesh: Default::default(),
            chunk_pos: *chunk_data.chunk_pos(),
            index
        };

        for i in 0..6 {
            render_info.chunk_mesh[i] = chunk_data.chunk_mesh(i as u32).cloned();

            if chunk_data.update_mesh(i as u32) {
                if let Some(mesh_data) = &mut chunk_data.updated_mesh_data[i] {
                    if let Some(chunk_mesh) = &mut chunk_data.chunk_mesh[i] {
                        self.upload_chunk_mesh_data(mesh_data, chunk_mesh)?;
                        // render_component.updated_mesh_data[i] = None;
                        chunk_data.notify_mesh_updated(i as u32);
                    }
                }
            }
        }

        self.render_info.push(render_info);

        self.object_data.push(object_data);
        self.object_indices.push(ObjectIndexUBO{ index });

        Ok(())
    }

    fn update_buffer_capacities(&mut self) {
        let num_objects = self.object_count;

        let growth_rate = 1.5;

        if num_objects > self.max_object_count {
            let prev_max_objects = self.max_object_count;
            self.max_object_count = util::grow_capacity(self.max_object_count, num_objects, growth_rate);
            debug!("VoxelRenderer - Growing ObjectData GPU buffers. Previous max objects: {prev_max_objects}, new max objects: {}", self.max_object_count);
        }

        let num_materials = self.materials.len() as u32;

        if num_materials > self.max_material_count {
            let prev_max_materials = self.max_material_count;
            self.max_material_count = util::grow_capacity(self.max_material_count, num_objects, growth_rate);
            debug!("VoxelRenderer - Growing Material GPU buffers. Previous max materials: {prev_max_materials}, new max materials: {}", self.max_material_count);
        }
    }

    fn update_object_date_transform(transform: &Transform, object_data_buffer: &mut ObjectDataUBO) {
        transform.write_model_matrix(&mut object_data_buffer.model_matrix)
    }

    fn build_draw_commands(&mut self, cmd_buf: &mut CommandBuffer, graphics: &GraphicsManager, view_pos: DVec3, frame_index: usize) -> Result<()> {

        let resource = &mut self.resources[frame_index];

        let view_chunk_pos = get_chunk_pos_for_world_pos(view_pos);

        self.chunk_buffer_handler.clear_indirect_draw_commands();

        let mut draw_command_count = 0;

        let vertex_size = size_of::<VoxelVertex>() as DeviceSize;
        let index_size = size_of::<u32>() as DeviceSize;

        for render_info in self.render_info.iter() {

            for i in 0..6 {
                if let Some(chunk_mesh) = render_info.chunk_mesh[i].as_ref() {
                    // let bounds = render_info.bounds[i].as_ref().unwrap();

                    if chunk_mesh.index_count == 0 {
                        // No vertices in this mesh, do not emit a command for it.
                        continue;
                    }

                    let axis = AxisDirection::from_index(i as u32).unwrap();

                    if render_info.chunk_pos != view_chunk_pos {
                        // This is not the chunk the viewer is within, therefor cull faces we cannot see.

                        let closest_point = closest_point_on_chunk(view_pos, render_info.chunk_pos);
                        // let closest_point = bounds.closest_point(view_pos);
                        let view_dir = view_pos - closest_point;

                        if view_dir.dot(axis.dvec()) < 0.0 {
                            continue; // This mesh is facing away from the viewer, cull it.
                        }
                    }

                    let vertex_buffer_region = &chunk_mesh.vertex_buffer_alloc;
                    let index_buffer_region = &chunk_mesh.index_buffer_alloc;

                    let indirect_draw_commands = self.chunk_buffer_handler.indirect_draw_commands(vertex_buffer_region.buffer_index, index_buffer_region.buffer_index);

                    let vertex_offset = (vertex_buffer_region.buffer_region.buffer_offset / vertex_size) as u32;
                    let first_index = (index_buffer_region.buffer_region.buffer_offset / index_size) as u32;

                    indirect_draw_commands.commands.push(DrawIndexedIndirectCommand{
                        index_count: chunk_mesh.index_count,
                        instance_count: 1,
                        first_index,
                        vertex_offset,
                        first_instance: render_info.index
                    });

                    draw_command_count += 1;

                    // resource.active_resources.push(chunk_mesh.vertex_buffer().buffer().clone());
                    // if let Some(index_buffer) = mesh.index_buffer() {
                    //     resource.active_resources.push(index_buffer.buffer().clone());
                    // }
                }
            }
        }

        if draw_command_count == 0 {
            // Nothing more to do
            return Ok(());
        }

        let growth_rate = 1.5;
        if draw_command_count > self.max_draw_commands_count {
            let prev_max_draw_commands = self.max_draw_commands_count;
            self.max_draw_commands_count = util::grow_capacity(self.max_draw_commands_count, draw_command_count, growth_rate);
            debug!("VoxelRenderer - Growing DrawCommand GPU buffers. Previous max commands: {prev_max_draw_commands}, new max objects: {}", self.max_draw_commands_count);
        }

        let allocator = graphics.memory_allocator();
        Self::map_indirect_draw_commands_buffer(resource, self.max_draw_commands_count as usize, allocator)?;

        let buffer_indirect_draw_commands = resource.buffer_indirect_draw_commands.as_mut().unwrap();
        debug_assert!(draw_command_count <= buffer_indirect_draw_commands.len() as u32);
        let mut writer = buffer_indirect_draw_commands.write()?;

        let mut idx_begin = 0;
        for (_key, commands) in self.chunk_buffer_handler.indirect_draw_commands.iter_mut() {
            let num_commands = commands.commands.len();
            let idx_end = idx_begin + num_commands;

            commands.indirect_buffer_offset = idx_begin;

            writer[idx_begin..idx_end].copy_from_slice(&commands.commands[0..num_commands]);
            idx_begin += commands.commands.len();
        }

        Ok(())
    }

    fn draw_scene(&mut self, cmd_buf: &mut CommandBuffer, scene: &mut Scene, view_pos: DVec3, frame_index: usize) -> Result<()> {

        let resource = &mut self.resources[frame_index];

        // if let Some(buffer_indirect_draw_commands) = &resource.buffer_indirect_draw_commands {
        //     for chunk_mesh_buffer in self.chunk_buffer_handler.buffers.iter() {
        //         cmd_buf.bind_vertex_buffers(0, chunk_mesh_buffer.buffer.clone())?;
        //         cmd_buf.bind_index_buffer_type(chunk_mesh_buffer.buffer.clone(), IndexType::U32)?;
        //
        //         cmd_buf.draw_indexed_indirect(buffer_indirect_draw_commands.clone())?;
        //     }
        //     resource.active_resources.push(buffer_indirect_draw_commands.buffer().clone());
        // }

        let mut curr_vertex_buffer_index: Option<usize> = None;
        let mut curr_index_buffer_index: Option<usize> = None;

        let mut count_bind_vertex_buffer = 0;
        let mut count_bind_index_buffer = 0;
        let mut count_draw_commands = 0;

        for (&(vertex_buffer_index, index_buffer_index), commands) in self.chunk_buffer_handler.indirect_draw_commands.iter() {

            if curr_vertex_buffer_index.is_none() || curr_vertex_buffer_index.unwrap() != vertex_buffer_index {
                curr_vertex_buffer_index = Some(vertex_buffer_index);

                let vertex_buffer = self.chunk_buffer_handler.get_gpu_buffer(vertex_buffer_index, ChunkMeshBufferType::VertexBuffer);
                cmd_buf.bind_vertex_buffers(0, vertex_buffer.clone())?;
                count_bind_vertex_buffer += 1;
            }

            if curr_index_buffer_index.is_none() || curr_index_buffer_index.unwrap() != index_buffer_index {
                curr_index_buffer_index = Some(index_buffer_index);

                let index_buffer = self.chunk_buffer_handler.get_gpu_buffer(index_buffer_index, ChunkMeshBufferType::IndexBuffer);
                cmd_buf.bind_index_buffer_type(index_buffer.clone(), IndexType::U32)?;
                count_bind_index_buffer += 1;
            }

            if let Some(indirect_buffer) = resource.buffer_indirect_draw_commands.as_ref() {
                if !commands.commands.is_empty() {
                    let slice_begin = commands.indirect_buffer_offset as DeviceSize;
                    let slice_end = slice_begin + commands.commands.len() as DeviceSize;

                    let indirect_buffer = indirect_buffer.clone().slice(slice_begin..slice_end);
                    cmd_buf.draw_indexed_indirect(indirect_buffer.clone())?;
                    count_draw_commands += 1;
                }
            }

            // for command in commands.iter() {
            //     cmd_buf.draw_indexed(command.index_count, command.instance_count, command.first_index, command.vertex_offset as i32, command.first_instance)?;
            //     count_draw_commands += 1;
            // }
        }




        // let view_chunk_pos = get_chunk_pos_for_world_pos(view_pos);
        //
        // let vertex_size = size_of::<VoxelVertex>() as DeviceSize;
        // let index_size = size_of::<u32>() as DeviceSize;
        //
        // for render_info in self.render_info.iter() {
        //     for i in 0..6 {
        //         if let Some(chunk_mesh) = render_info.chunk_mesh[i].as_ref() {
        //             // let bounds = render_info.bounds[i].as_ref().unwrap();
        //
        //             let axis = AxisDirection::from_index(i as u32).unwrap();
        //
        //             if render_info.chunk_pos != view_chunk_pos {
        //                 // This is not the chunk the viewer is within, therefor cull faces we cannot see.
        //
        //                 let closest_point = closest_point_on_chunk(view_pos, render_info.chunk_pos);
        //                 // let closest_point = bounds.closest_point(view_pos);
        //                 let view_dir = view_pos - closest_point;
        //
        //                 if view_dir.dot(axis.dvec()) < 0.0 {
        //                     continue; // This mesh is facing away from the viewer, cull it.
        //                 }
        //             }
        //
        //             let vertex_buffer_region = &chunk_mesh.vertex_buffer_alloc;
        //             let index_buffer_region = &chunk_mesh.index_buffer_alloc;
        //
        //             if curr_vertex_buffer_index.is_none() || curr_vertex_buffer_index.unwrap() != vertex_buffer_region.buffer_index {
        //                 curr_vertex_buffer_index = Some(vertex_buffer_region.buffer_index);
        //
        //                 let vertex_buffer = self.chunk_buffer_handler.get_gpu_buffer(vertex_buffer_region.buffer_index, ChunkMeshBufferType::VertexBuffer);
        //                 cmd_buf.bind_vertex_buffers(0, vertex_buffer.clone())?;
        //                 count_bind_vertex_buffer += 1;
        //             }
        //
        //             if curr_index_buffer_index.is_none() || curr_index_buffer_index.unwrap() != index_buffer_region.buffer_index {
        //                 curr_index_buffer_index = Some(index_buffer_region.buffer_index);
        //
        //                 let index_buffer = self.chunk_buffer_handler.get_gpu_buffer(index_buffer_region.buffer_index, ChunkMeshBufferType::IndexBuffer);
        //                 cmd_buf.bind_index_buffer_type(index_buffer.clone(), IndexType::U32)?;
        //                 count_bind_index_buffer += 1;
        //             }
        //
        //             let first_vertex = (vertex_buffer_region.buffer_region.buffer_offset / vertex_size) as i32;
        //             let first_index = (index_buffer_region.buffer_region.buffer_offset / index_size) as u32;
        //
        //             cmd_buf.draw_indexed(chunk_mesh.index_count, 1, first_index, first_vertex, render_info.index)?;
        //             count_draw_commands += 1;
        //         }
        //     }
        // }



        // debug!("Ran {count_draw_commands} draw commands, bound {count_bind_vertex_buffer} vertex buffers, bound {count_bind_index_buffer} index buffers");

        Ok(())
    }

    fn update_gpu_resources(&mut self, engine: &Engine) -> Result<()> {

        let frame_index = engine.graphics.current_frame_index();
        let resource = &mut self.resources[frame_index];

        Self::update_object_data_gpu_resources(self.object_count as usize, &self.object_data, &self.object_indices, resource)?;

        Self::update_material_data_gpu_resources(&self.textures, &self.material_data, self.null_texture.clone(), self.default_sampler.clone(), resource)?;

        Ok(())
    }

    fn update_object_data_gpu_resources(object_count: usize, object_data: &[ObjectDataUBO], object_indices: &[ObjectIndexUBO], resource: &mut FrameResource) -> Result<()> {

        if !resource.descriptor_writes_world.is_empty() {
            let descriptor_set_world = resource.descriptor_set_world.as_ref().unwrap().clone();
            let writes = mem::take(&mut resource.descriptor_writes_world);

            unsafe { descriptor_set_world.update_by_ref(writes, []) }?
        }

        let static_begin = 0;
        let static_end = static_begin + object_count;

        if object_count > 0 {

            {
                let buffer_object_data = resource.buffer_object_data.as_mut().unwrap();
                debug_assert!(object_data.len() <= buffer_object_data.len() as usize);
                let mut writer = buffer_object_data.write()?;

                if resource.scene_changed && object_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&object_data[static_begin..static_end]);
                }
            }


            {
                let buffer_object_indices = resource.buffer_object_indices.as_mut().unwrap();
                debug_assert!(object_indices.len() <= buffer_object_indices.len() as usize);
                let mut writer = buffer_object_indices.write()?;

                if resource.scene_changed && object_count > 0 {
                    writer[static_begin..static_end].copy_from_slice(&object_indices[static_begin..static_end]);
                }
            }
        }

        resource.scene_changed = false;

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


    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        debug!("VoxelRenderer - Recreate swapchain");
        self.create_main_graphics_pipeline(engine)?;

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
        info!("Initialize VoxelRenderer GraphicsPipeline");
        let device = engine.graphics.device();
        let render_pass = engine.graphics.render_pass();

        let mut shader_source = String::new();
        let mut file = File::open("./res/shaders/voxel.glsl")?;
        file.read_to_string(&mut shader_source)?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
        options.add_macro_definition("MASK_X", Some(format!("{}", VoxelVertex::MASK_X).as_str()));
        options.add_macro_definition("SHIFT_X", Some(format!("{}", VoxelVertex::SHIFT_X).as_str()));
        options.add_macro_definition("MASK_Y", Some(format!("{}", VoxelVertex::MASK_Y).as_str()));
        options.add_macro_definition("SHIFT_Y", Some(format!("{}", VoxelVertex::SHIFT_Y).as_str()));
        options.add_macro_definition("MASK_Z", Some(format!("{}", VoxelVertex::MASK_Z).as_str()));
        options.add_macro_definition("SHIFT_Z", Some(format!("{}", VoxelVertex::SHIFT_Z).as_str()));
        options.add_macro_definition("MASK_NORM", Some(format!("{}", VoxelVertex::MASK_NORM).as_str()));
        options.add_macro_definition("SHIFT_NORM", Some(format!("{}", VoxelVertex::SHIFT_NORM).as_str()));
        options.add_macro_definition("VERTEX_SHADER_MODULE", None);
        let vs = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "voxel.glsl::vert", "main", ShaderKind::Vertex, Some(&options))?;

        let mut options = CompileOptions::new()?;
        options.add_macro_definition("WORLD_SCALE", Some(Transform::WORLD_SCALE.to_string().as_str()));
        options.add_macro_definition("FRAGMENT_SHADER_MODULE", None);
        let fs_solid = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "voxel.glsl::frag(solid)", "main", ShaderKind::Fragment, Some(&options))?;

        options.add_macro_definition("WIREFRAME_ENABLED", None);
        let fs_wire = engine.graphics.load_shader_module_from_source(shader_source.as_str(), "voxel.glsl::frag(wire)", "main", ShaderKind::Fragment, Some(&options))?;

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
            .build_pipeline::<VoxelVertex>(device.clone())?;

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
            .build_pipeline::<VoxelVertex>(device.clone())?;


        self.solid_graphics_pipeline = Some(main_graphics_pipeline);
        self.wire_graphics_pipeline = Some(wire_graphics_pipeline);

        Ok(())
    }

    fn create_camera_descriptor_sets(camera: &RenderCamera, resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {

        let descriptor_set = camera.create_camera_descriptor_sets(0, 0, graphics_pipeline, graphics)?;
        resource.descriptor_set_camera = Some(descriptor_set);

        Ok(())
    }

    fn create_main_descriptor_sets(resource: &mut FrameResource, graphics_pipeline: &GraphicsPipeline, graphics: &GraphicsManager) -> Result<()> {
        debug!("VoxelRenderer - create_main_descriptor_sets");

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

    fn map_indirect_draw_commands_buffer(resource: &mut FrameResource, max_commands_count: usize, memory_allocator: Arc<dyn MemoryAllocator>) -> Result<()>{
        let max_buffer_size: DeviceSize = (size_of::<DrawIndexedIndirectCommand>() * max_commands_count) as DeviceSize;

        if let Some(buffer) = &resource.buffer_indirect_draw_commands {
            if max_buffer_size > buffer.size() {
                // Existing buffer is too small. It will be re-allocated
                resource.buffer_indirect_draw_commands = None;
            }
        }

        if resource.buffer_indirect_draw_commands.is_none() {
            debug!("Allocating IndirectCommands GPU buffer for {max_commands_count} commands ({max_buffer_size} bytes)");

            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buffer_indirect_draw_commands = Buffer::new_slice::<DrawIndexedIndirectCommand>(memory_allocator.clone(), buffer_create_info, allocation_info, max_commands_count as DeviceSize)?;

            resource.buffer_indirect_draw_commands = Some(buffer_indirect_draw_commands);
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
        set_vulkan_debug_name(staging_buffer.buffer(), Some("VoxelRenderer-CreateNullTexture-StagingBuffer"))?;
        let image_view = Texture::create_image_view_2d(allocator, width, height, Format::R8G8B8A8_UNORM, ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST, Some("VoxelRenderer-NullTexture"))?;

        Texture::load_image_from_data_staged(&mut cmd_buf, &staging_buffer, &data, image_view.image().clone())?;

        graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;
        
        Ok(image_view)
    }

    fn create_default_sampler(device: Arc<Device>) -> Result<Arc<Sampler>> {
        Texture::create_default_sampler(device)
    }

    pub fn draw_debug_gui(&self, ticker: &mut Ticker, ctx: &Context) {
        self.chunk_buffer_handler.draw_debug_gui(ticker, ctx);
    }
    // pub fn add_mesh(&mut self, mesh: Mesh<BaseVertex>) {
    //     self.meshes.push(mesh);
    // }
}