use std::ops::Range;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use smallvec::SmallVec;
use vulkano::buffer::IndexBuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SecondaryAutoCommandBuffer, SubpassBeginInfo, SubpassEndInfo};
use vulkano::descriptor_set::DescriptorSetsCollection;
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint, PipelineLayout};
use vulkano::query::{QueryControlFlags, QueryPool};
use vulkano::sync::PipelineStage;

type PrimaryCommandBuffer = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;
type SecondaryCommandBuffer = AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>;


pub enum CommandBufferType {
    Primary(PrimaryCommandBuffer),
    Secondary(SecondaryCommandBuffer),
}

pub struct CommandBuffer {
    cmd_buf: CommandBufferType
}

impl CommandBuffer {
    pub fn new(cmd_buf: CommandBufferType) -> Self {
        CommandBuffer {
            cmd_buf
        }
    }
    
    pub fn get(&self) -> &CommandBufferType {
        &self.cmd_buf
    }
    
    pub fn end_query(&mut self, query_pool: Arc<QueryPool>, query: u32) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.end_query(query_pool, query)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.end_query(query_pool, query)?; }
        };
        Ok(())
    }
    
    pub fn begin_query(&mut self, query_pool: Arc<QueryPool>, query: u32, flags: QueryControlFlags) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.begin_query(query_pool, query, flags) }?; },
            CommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.begin_query(query_pool, query, flags) }?; }
        };
        Ok(())
    }
    
    pub fn reset_query_pool(&mut self, query_pool: Arc<QueryPool>, queries: Range<u32>) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.reset_query_pool(query_pool, queries) }?; }
            CommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.reset_query_pool(query_pool, queries) }?; }
        };
        Ok(())
    }
    
    pub fn write_timestamp(&mut self, query_pool: Arc<QueryPool>, query: u32, stage: PipelineStage) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.write_timestamp(query_pool, query, stage) }?; },
            CommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.write_timestamp(query_pool, query, stage) }?; }
        };
        Ok(())
    }
    
    pub fn copy_buffer_to_image(&mut self, copy_buffer_to_image_info: CopyBufferToImageInfo) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.copy_buffer_to_image(copy_buffer_to_image_info)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.copy_buffer_to_image(copy_buffer_to_image_info)?; }
        };
        Ok(())
    }
    
    pub fn copy_buffer(&mut self, copy_buffer_info: CopyBufferInfo) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.copy_buffer(copy_buffer_info)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.copy_buffer(copy_buffer_info)?; }
        };
        Ok(())
    }
    
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, vertex_buffers: impl VertexBuffersCollection) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_vertex_buffers(first_binding, vertex_buffers)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_vertex_buffers(first_binding, vertex_buffers)?; }
        };
        Ok(())
    }
    
    pub fn bind_index_buffer(&mut self, index_buffer: impl Into<IndexBuffer>) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_index_buffer(index_buffer)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_index_buffer(index_buffer)?; }
        };
        Ok(())
    }
    
    pub fn draw_indexed(&mut self, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance) }?; }
            CommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.draw_indexed(index_count, instance_count, first_index, vertex_offset, first_instance) }?; }
        };
        Ok(())
    }
    
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { unsafe { cmd_buf.draw(vertex_count, instance_count, first_vertex, first_instance) }?; }
            CommandBufferType::Secondary(cmd_buf) => { unsafe { cmd_buf.draw(vertex_count, instance_count, first_vertex, first_instance) }?; }
        };
        Ok(())
    }

    pub fn begin_render_pass(&mut self, render_pass_begin_info: RenderPassBeginInfo, subpass_begin_info: SubpassBeginInfo) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.begin_render_pass(render_pass_begin_info, subpass_begin_info)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.begin_render_pass(render_pass_begin_info, subpass_begin_info)?; }
        };
        Ok(())
    }

    pub fn end_render_pass(&mut self, subpass_end_info: SubpassEndInfo) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.end_render_pass(subpass_end_info)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.end_render_pass(subpass_end_info)?; }
        };
        Ok(())
    }

    pub fn set_viewport(&mut self, first_viewport: u32, viewports: SmallVec<[Viewport; 2]>) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.set_viewport(first_viewport, viewports)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.set_viewport(first_viewport, viewports)?; }
        };
        Ok(())
    }

    pub fn bind_descriptor_sets(&mut self, pipeline_bind_point: PipelineBindPoint, pipeline_layout: Arc<PipelineLayout>, first_set: u32, descriptor_sets: impl DescriptorSetsCollection) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_descriptor_sets(pipeline_bind_point, pipeline_layout, first_set, descriptor_sets)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_descriptor_sets(pipeline_bind_point, pipeline_layout, first_set, descriptor_sets)?; }
        };
        Ok(())
    }

    pub fn bind_pipeline_graphics(&mut self, pipeline: Arc<GraphicsPipeline>) -> Result<()> {
        match &mut self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => { cmd_buf.bind_pipeline_graphics(pipeline)?; },
            CommandBufferType::Secondary(cmd_buf) => { cmd_buf.bind_pipeline_graphics(pipeline)?; }
        };
        Ok(())
    }
    
    pub fn build_primary(self) -> Result<Arc<PrimaryAutoCommandBuffer>>{
        match self.cmd_buf {
            CommandBufferType::Primary(cmd_buf) => Ok(cmd_buf.build()?),
            CommandBufferType::Secondary(cmd_buf) => Err(anyhow!("Cannot build SecondaryCommandBuffer as primary"))
        }
    }
}


pub trait CommandBufferImpl {
    
}

impl CommandBufferImpl for CommandBuffer {
    
}