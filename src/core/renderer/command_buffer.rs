use std::ops::Range;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use ash::{vk};
use ash::vk::{CommandBufferUsageFlags, Extent2D, Extent3D, ImageSubresourceLayers, Offset2D, Offset3D, Rect2D};
use smallvec::{ExtendFromSlice, SmallVec};
use vulkano::buffer::IndexBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassBeginInfo, SubpassEndInfo};
use vulkano::descriptor_set::DescriptorSetsCollection;
use vulkano::format::ClearValue;
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint, PipelineLayout};
use vulkano::query::{QueryControlFlags, QueryPool};
use vulkano::sync::{BufferMemoryBarrier, DependencyFlags, ImageMemoryBarrier, MemoryBarrier, PipelineStage, PipelineStages, QueueFamilyOwnershipTransfer};
use vulkano::VulkanObject;

type PrimaryCommandBuffer = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;
type SecondaryCommandBuffer = AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>;


// pub type CommandBuffer = VulkanoCommandBuffer;
pub type CommandBuffer = AshCommandBuffer;


pub trait CommandBufferImpl {
    fn begin(&mut self, usage: CommandBufferUsage) -> Result<()>;

    fn end(&mut self) -> Result<()>;

    fn end_query(&mut self, query_pool: Arc<QueryPool>, query: u32) -> Result<()>;

    fn begin_query(&mut self, query_pool: Arc<QueryPool>, query: u32, flags: QueryControlFlags) -> Result<()>;

    fn reset_query_pool(&mut self, query_pool: Arc<QueryPool>, queries: Range<u32>) -> Result<()>;

    fn write_timestamp(&mut self, query_pool: Arc<QueryPool>, query: u32, stage: PipelineStage) -> Result<()>;

    fn copy_buffer_to_image(&mut self, copy_buffer_to_image_info: CopyBufferToImageInfo) -> Result<()>;

    fn copy_buffer(&mut self, copy_buffer_info: CopyBufferInfo) -> Result<()>;

    fn bind_vertex_buffers(&mut self, first_binding: u32, vertex_buffers: impl VertexBuffersCollection) -> Result<()>;

    fn bind_index_buffer(&mut self, index_buffer: impl Into<IndexBuffer>) -> Result<()>;

    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<()>;

    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<()>;

    fn begin_render_pass(&mut self, render_pass_begin_info: RenderPassBeginInfo, subpass_begin_info: SubpassBeginInfo) -> Result<()>;

    fn end_render_pass(&mut self, subpass_end_info: SubpassEndInfo) -> Result<()>;

    fn set_viewport(&mut self, first_viewport: u32, viewports: SmallVec<[Viewport; 2]>) -> Result<()>;

    fn bind_descriptor_sets(&mut self, pipeline_bind_point: PipelineBindPoint, pipeline_layout: Arc<PipelineLayout>, first_set: u32, descriptor_sets: impl DescriptorSetsCollection) -> Result<()>;

    fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> Result<()>;

    fn pipeline_barrier(&mut self, src_stage: PipelineStages, dst_stage: PipelineStages, dependency_flags: DependencyFlags, memory_barriers: &[MemoryBarrier], buffer_memory_barriers: &[BufferMemoryBarrier], image_memory_barriers: &[ImageMemoryBarrier]) -> Result<()>;
}


pub enum VulkanoCommandBufferType {
    Primary(PrimaryCommandBuffer),
    Secondary(SecondaryCommandBuffer),
}

pub struct VulkanoCommandBuffer {
    cmd_buf: VulkanoCommandBufferType,
}

impl VulkanoCommandBuffer {
    pub fn new(cmd_buf: VulkanoCommandBufferType) -> Self {
        VulkanoCommandBuffer {
            cmd_buf,
        }
    }

    pub fn get(&self) -> &VulkanoCommandBufferType {
        &self.cmd_buf
    }

    pub fn build_primary(self) -> Result<Arc<PrimaryAutoCommandBuffer>>{
        match self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => Ok(cmd_buf.build()?),
            VulkanoCommandBufferType::Secondary(cmd_buf) => Err(anyhow!("Cannot build SecondaryCommandBuffer as primary"))
        }
    }
}


impl CommandBufferImpl for VulkanoCommandBuffer {
    fn begin(&mut self, _usage: CommandBufferUsage) -> Result<()> {
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        Ok(())
    }

    fn end_query(&mut self, query_pool: Arc<QueryPool>, query: u32) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.end_query(query_pool, query)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.end_query(query_pool, query)?; }
        };
        Ok(())
    }

    fn begin_query(&mut self, query_pool: Arc<QueryPool>, query: u32, flags: QueryControlFlags) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.begin_query(query_pool, query, flags) }?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.begin_query(query_pool, query, flags) }?; }
        };
        Ok(())
    }

    fn reset_query_pool(&mut self, query_pool: Arc<QueryPool>, queries: Range<u32>) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.reset_query_pool(query_pool, queries) }?; }
            VulkanoCommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.reset_query_pool(query_pool, queries) }?; }
        };
        Ok(())
    }

    fn write_timestamp(&mut self, query_pool: Arc<QueryPool>, query: u32, stage: PipelineStage) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.write_timestamp(query_pool, query, stage) }?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.write_timestamp(query_pool, query, stage) }?; }
        };
        Ok(())
    }

    fn copy_buffer_to_image(&mut self, copy_buffer_to_image_info: CopyBufferToImageInfo) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.copy_buffer_to_image(copy_buffer_to_image_info)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.copy_buffer_to_image(copy_buffer_to_image_info)?; }
        };
        Ok(())
    }

    fn copy_buffer(&mut self, copy_buffer_info: CopyBufferInfo) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.copy_buffer(copy_buffer_info)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.copy_buffer(copy_buffer_info)?; }
        };
        Ok(())
    }

    fn bind_vertex_buffers(&mut self, first_binding: u32, vertex_buffers: impl VertexBuffersCollection) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_vertex_buffers(first_binding, vertex_buffers)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_vertex_buffers(first_binding, vertex_buffers)?; }
        };
        Ok(())
    }

    fn bind_index_buffer(&mut self, index_buffer: impl Into<IndexBuffer>) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_index_buffer(index_buffer)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_index_buffer(index_buffer)?; }
        };
        Ok(())
    }

    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance) }?; }
            VulkanoCommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance) }?; }
        };
        Ok(())
    }

    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.draw(vertex_count, instance_count, first_vertex, first_instance) }?; }
            VulkanoCommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.draw(vertex_count, instance_count, first_vertex, first_instance) }?; }
        };
        Ok(())
    }


    fn begin_render_pass(&mut self, render_pass_begin_info: RenderPassBeginInfo, subpass_begin_info: SubpassBeginInfo) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.begin_render_pass(render_pass_begin_info, subpass_begin_info)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.begin_render_pass(render_pass_begin_info, subpass_begin_info)?; }
        };
        Ok(())
    }

    fn end_render_pass(&mut self, subpass_end_info: SubpassEndInfo) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.end_render_pass(subpass_end_info)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.end_render_pass(subpass_end_info)?; }
        };
        Ok(())
    }

    fn set_viewport(&mut self, first_viewport: u32, viewports: SmallVec<[Viewport; 2]>) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.set_viewport(first_viewport, viewports)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.set_viewport(first_viewport, viewports)?; }
        };
        Ok(())
    }

    fn bind_descriptor_sets(&mut self, pipeline_bind_point: PipelineBindPoint, pipeline_layout: Arc<PipelineLayout>, first_set: u32, descriptor_sets: impl DescriptorSetsCollection) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_descriptor_sets(pipeline_bind_point, pipeline_layout, first_set, descriptor_sets)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_descriptor_sets(pipeline_bind_point, pipeline_layout, first_set, descriptor_sets)?; }
        };
        Ok(())
    }

    fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> Result<()> {
        match &mut self.cmd_buf {
            VulkanoCommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_pipeline_graphics(pipeline)?; },
            VulkanoCommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_pipeline_graphics(pipeline)?; }
        };
        Ok(())
    }

    fn pipeline_barrier(&mut self, src_stage: PipelineStages, dst_stage: PipelineStages, dependency_flags: DependencyFlags, memory_barriers: &[MemoryBarrier], buffer_memory_barriers: &[BufferMemoryBarrier], image_memory_barriers: &[ImageMemoryBarrier]) -> Result<()> {
        // Vulkano handles this for us. Nothing to do.
        Ok(())
    }
}


#[derive(Clone)]
pub struct AshCommandBuffer {
    device: ash::Device,
    cmd_buf: vk::CommandBuffer,
    debug_draw_commands: u32,
}

impl AshCommandBuffer {
    pub fn new(device: ash::Device, command_pool: vk::CommandPool, level: vk::CommandBufferLevel) -> Self {

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .level(level);

        let cmd_buf = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0];

        AshCommandBuffer {
            device,
            cmd_buf,
            debug_draw_commands: 0,
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn handle(&self) -> &vk::CommandBuffer {
        &self.cmd_buf
    }

    pub fn debug_draw_commands(&self) -> u32 {
        self.debug_draw_commands
    }
}

impl CommandBufferImpl for AshCommandBuffer {
    fn begin(&mut self, usage: CommandBufferUsage) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(self.cmd_buf, &begin_info) }?;
        self.debug_draw_commands = 0;
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        unsafe { self.device.end_command_buffer(self.cmd_buf) }?;
        Ok(())
    }

    fn end_query(&mut self, query_pool: Arc<QueryPool>, query: u32) -> Result<()> {
        unsafe { self.device.cmd_end_query(self.cmd_buf, query_pool.handle(), query) };
        Ok(())
    }

    fn begin_query(&mut self, query_pool: Arc<QueryPool>, query: u32, flags: QueryControlFlags) -> Result<()> {
        unsafe { self.device.cmd_begin_query(self.cmd_buf, query_pool.handle(), query, flags.into()) };
        Ok(())
    }

    fn reset_query_pool(&mut self, query_pool: Arc<QueryPool>, queries: Range<u32>) -> Result<()> {
        unsafe { self.device.cmd_reset_query_pool(self.cmd_buf, query_pool.handle(), queries.start, queries.end - queries.start) };
        Ok(())
    }

    fn write_timestamp(&mut self, query_pool: Arc<QueryPool>, query: u32, stage: PipelineStage) -> Result<()> {
        unsafe { self.device.cmd_write_timestamp(self.cmd_buf, stage.into(), query_pool.handle(), query) };
        Ok(())
    }

    fn copy_buffer_to_image(&mut self, copy_buffer_to_image_info: CopyBufferToImageInfo) -> Result<()> {
        let src_buffer = copy_buffer_to_image_info.src_buffer.buffer().handle();
        let dst_image = copy_buffer_to_image_info.dst_image.handle();
        let dst_image_layout = copy_buffer_to_image_info.dst_image_layout.into();
        let mut regions = Vec::with_capacity(copy_buffer_to_image_info.regions.len());
        for (i, image_copy) in copy_buffer_to_image_info.regions.iter().enumerate() {
            regions.push(vk::BufferImageCopy::default()
                .buffer_offset(copy_buffer_to_image_info.src_buffer.offset() + image_copy.buffer_offset)
                .buffer_row_length(image_copy.buffer_row_length)
                .buffer_image_height(image_copy.buffer_image_height)
                .image_subresource(ImageSubresourceLayers::default()
                    .aspect_mask(image_copy.image_subresource.aspects.into())
                    .mip_level(image_copy.image_subresource.mip_level)
                    .base_array_layer(image_copy.image_subresource.array_layers.start)
                    .layer_count(image_copy.image_subresource.array_layers.end - image_copy.image_subresource.array_layers.start))
                .image_offset(Offset3D{ x: image_copy.image_offset[0] as i32, y: image_copy.image_offset[1] as i32, z: image_copy.image_offset[2] as i32 })
                .image_extent(Extent3D{ width: image_copy.image_extent[0], height: image_copy.image_extent[1], depth: image_copy.image_extent[2] }));
        }
        unsafe { self.device.cmd_copy_buffer_to_image(self.cmd_buf, src_buffer, dst_image, dst_image_layout, &regions) };
        Ok(())
    }

    fn copy_buffer(&mut self, copy_buffer_info: CopyBufferInfo) -> Result<()> {
        let src_buffer = copy_buffer_info.src_buffer.buffer().handle();
        let dst_buffer = copy_buffer_info.dst_buffer.buffer().handle();
        let mut regions = Vec::with_capacity(copy_buffer_info.regions.len());
        for (i, buffer_copy) in copy_buffer_info.regions.iter().enumerate() {
            regions.push(vk::BufferCopy::default()
                .src_offset(copy_buffer_info.src_buffer.offset() + buffer_copy.src_offset)
                .dst_offset(copy_buffer_info.dst_buffer.offset() + buffer_copy.dst_offset)
                .size(buffer_copy.size));
        }
        unsafe { self.device.cmd_copy_buffer(self.cmd_buf, src_buffer, dst_buffer, &regions) };
        Ok(())
    }

    fn bind_vertex_buffers(&mut self, first_binding: u32, vertex_buffers: impl VertexBuffersCollection) -> Result<()> {
        let vertex_buffers = vertex_buffers.into_vec();
        let mut buffers = Vec::with_capacity(vertex_buffers.len());
        let mut offsets = Vec::with_capacity(vertex_buffers.len());
        for (i, buffer) in vertex_buffers.iter().enumerate() {
            buffers.push(buffer.buffer().handle());
            offsets.push(buffer.offset());
        }
        unsafe { self.device.cmd_bind_vertex_buffers(self.cmd_buf, first_binding, &buffers, &offsets) };
        Ok(())
    }

    fn bind_index_buffer(&mut self, index_buffer: impl Into<IndexBuffer>) -> Result<()> {
        let index_buffer = index_buffer.into();
        let buffer = index_buffer.as_bytes();
        let offset = buffer.offset();
        let index_type = index_buffer.index_type().into();
        unsafe { self.device.cmd_bind_index_buffer(self.cmd_buf, buffer.buffer().handle(), offset, index_type) };
        Ok(())
    }

    fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<()> {
        unsafe { self.device.cmd_draw_indexed(self.cmd_buf, index_count, instance_count, first_index, vertex_offset, first_instance) };
        self.debug_draw_commands += 1;
        Ok(())
    }

    fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<()> {
        unsafe { self.device.cmd_draw(self.cmd_buf, vertex_count, instance_count, first_vertex, first_instance) };
        self.debug_draw_commands += 1;
        Ok(())
    }

    fn begin_render_pass(&mut self, render_pass_begin_info: RenderPassBeginInfo, subpass_begin_info: SubpassBeginInfo) -> Result<()> {
        let mut clear_values = vec![vk::ClearValue::default(); render_pass_begin_info.clear_values.len()];
        for (i, clear_value) in render_pass_begin_info.clear_values.iter().enumerate() {
            if let Some(clear_value) = clear_value {
                unsafe {
                    match clear_value {
                        ClearValue::Float(val) => clear_values[i].color.float32 = *val,
                        ClearValue::Int(val) => clear_values[i].color.int32 = *val,
                        ClearValue::Uint(val) => clear_values[i].color.uint32 = *val,
                        ClearValue::Depth(val) => clear_values[i].depth_stencil.depth = *val,
                        ClearValue::Stencil(val) => clear_values[i].depth_stencil.stencil = *val,
                        ClearValue::DepthStencil((depth, stencil)) => clear_values[i].depth_stencil = vk::ClearDepthStencilValue { depth: *depth, stencil: *stencil }
                    };
                }
            }
        }
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass_begin_info.render_pass.handle())
            .framebuffer(render_pass_begin_info.framebuffer.handle())
            .render_area(Rect2D{
                offset: Offset2D{ x: render_pass_begin_info.render_area_offset[0] as i32, y: render_pass_begin_info.render_area_offset[1] as i32 },
                extent: Extent2D{ width: render_pass_begin_info.render_area_extent[0], height: render_pass_begin_info.render_area_extent[1]}
            })
            .clear_values(&clear_values);

        unsafe { self.device.cmd_begin_render_pass(self.cmd_buf, &render_pass_begin, subpass_begin_info.contents.into()) };
        Ok(())
    }

    fn end_render_pass(&mut self, _subpass_end_info: SubpassEndInfo) -> Result<()> {
        unsafe { self.device.cmd_end_render_pass(self.cmd_buf) };
        Ok(())
    }

    fn set_viewport(&mut self, first_viewport: u32, viewports: SmallVec<[Viewport; 2]>) -> Result<()> {
        let mut viewports1 = Vec::with_capacity(viewports.len());
        for (i, viewport) in viewports.iter().enumerate() {
            viewports1.push(vk::Viewport{ x: viewport.offset[0], y: viewport.offset[1], width: viewport.extent[0], height: viewport.extent[1], min_depth: *viewport.depth_range.start(), max_depth: *viewport.depth_range.end() });
        }
        unsafe { self.device.cmd_set_viewport(self.cmd_buf, first_viewport, &viewports1) };
        Ok(())
    }

    fn bind_descriptor_sets(&mut self, pipeline_bind_point: PipelineBindPoint, pipeline_layout: Arc<PipelineLayout>, first_set: u32, descriptor_sets: impl DescriptorSetsCollection) -> Result<()> {
        let descriptor_sets_list = descriptor_sets.into_vec();
        let mut descriptor_sets = Vec::with_capacity(descriptor_sets_list.len());
        let mut dynamic_offsets = vec![];

        for (i, descriptor_set_with_offset) in descriptor_sets_list.iter().enumerate() {
            let (curr_descriptor_set, curr_dynamic_offsets) = descriptor_set_with_offset.as_ref();
            descriptor_sets.push(curr_descriptor_set.handle());
            dynamic_offsets.extend_from_slice(curr_dynamic_offsets);
        }

        unsafe { self.device.cmd_bind_descriptor_sets(self.cmd_buf, pipeline_bind_point.into(), pipeline_layout.handle(), first_set, &descriptor_sets, &dynamic_offsets) };
        Ok(())
    }

    fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> Result<()> {
        let pipeline = pipeline.handle();
        unsafe { self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline) };
        Ok(())
    }

    fn pipeline_barrier(&mut self, src_stage: PipelineStages, dst_stage: PipelineStages, dependency_flags: DependencyFlags, memory_barriers: &[MemoryBarrier], buffer_memory_barriers: &[BufferMemoryBarrier], image_memory_barriers: &[ImageMemoryBarrier]) -> Result<()>{
        let dependency_flags = dependency_flags.into();
        let src_stage_mask = src_stage.into();
        let dst_stage_mask = dst_stage.into();
        let mut vk_memory_barriers = Vec::with_capacity(memory_barriers.len());
        let mut vk_buffer_memory_barriers = Vec::with_capacity(buffer_memory_barriers.len());
        let mut vk_image_memory_barriers = Vec::with_capacity(image_memory_barriers.len());

        for (i, memory_barrier) in memory_barriers.iter().enumerate() {
            vk_memory_barriers.push(vk::MemoryBarrier::default()
                .src_access_mask(memory_barrier.src_access.into())
                .dst_access_mask(memory_barrier.dst_access.into()));
        }
        for (i, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            let (src_queue_family_index, dst_queue_family_index) = buffer_memory_barrier.queue_family_ownership_transfer.as_ref()
                .map_or((vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED), queue_family_ownership_transfer_to_vk);
            vk_buffer_memory_barriers.push(vk::BufferMemoryBarrier::default()
                .src_access_mask(buffer_memory_barrier.src_access.into())
                .dst_access_mask(buffer_memory_barrier.dst_access.into())
                .src_queue_family_index(src_queue_family_index)
                .dst_queue_family_index(dst_queue_family_index)
                .buffer(buffer_memory_barrier.buffer.handle())
                .offset(buffer_memory_barrier.range.start)
                .size(buffer_memory_barrier.range.end - buffer_memory_barrier.range.start));
        }
        for (i, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            let (src_queue_family_index, dst_queue_family_index) = image_memory_barrier.queue_family_ownership_transfer.as_ref()
                .map_or((vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED), queue_family_ownership_transfer_to_vk);
            vk_image_memory_barriers.push(vk::ImageMemoryBarrier::default()
                .src_access_mask(image_memory_barrier.src_access.into())
                .dst_access_mask(image_memory_barrier.dst_access.into())
                .old_layout(image_memory_barrier.old_layout.into())
                .new_layout(image_memory_barrier.new_layout.into())
                .src_queue_family_index(src_queue_family_index)
                .dst_queue_family_index(dst_queue_family_index)
                .image(image_memory_barrier.image.handle())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: image_memory_barrier.subresource_range.aspects.into(),
                    base_mip_level: image_memory_barrier.subresource_range.mip_levels.start,
                    level_count: image_memory_barrier.subresource_range.mip_levels.end - image_memory_barrier.subresource_range.mip_levels.start,
                    base_array_layer: image_memory_barrier.subresource_range.array_layers.start,
                    layer_count: image_memory_barrier.subresource_range.array_layers.end - image_memory_barrier.subresource_range.array_layers.start,
                }));
        }

        unsafe { self.device.cmd_pipeline_barrier(self.cmd_buf, src_stage_mask, dst_stage_mask, dependency_flags, &vk_memory_barriers, &vk_buffer_memory_barriers, &vk_image_memory_barriers) }
        Ok(())
    }
}


fn queue_family_ownership_transfer_to_vk(transfer: &QueueFamilyOwnershipTransfer) -> (u32, u32) {
    match *transfer {
        QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
            src_index,
            dst_index,
        } => (src_index, dst_index),
        QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index } => (src_index, vk::QUEUE_FAMILY_EXTERNAL),
        QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index } => {
            (vk::QUEUE_FAMILY_EXTERNAL, dst_index)
        }
        QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index } => {
            (src_index, vk::QUEUE_FAMILY_FOREIGN_EXT)
        }
        QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index } => {
            (vk::QUEUE_FAMILY_FOREIGN_EXT, dst_index)
        }
        QueueFamilyOwnershipTransfer::ConcurrentToExternal => (
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_EXTERNAL,
        ),
        QueueFamilyOwnershipTransfer::ConcurrentFromExternal => (
            vk::QUEUE_FAMILY_EXTERNAL,
            vk::QUEUE_FAMILY_IGNORED,
        ),
        QueueFamilyOwnershipTransfer::ConcurrentToForeign => (
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_FOREIGN_EXT,
        ),
        QueueFamilyOwnershipTransfer::ConcurrentFromForeign => (
            vk::QUEUE_FAMILY_FOREIGN_EXT,
            vk::QUEUE_FAMILY_IGNORED,
        ),
    }
}