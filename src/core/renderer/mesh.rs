use crate::core::PrimaryCommandBuffer;
use anyhow::Result;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;

pub struct MeshConfiguration<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Option<Vec<u32>>,
}



pub struct Mesh<V: Vertex> {
    vertex_buffer: Subbuffer<[V]>,
    index_buffer: Option<Subbuffer<[u32]>>,

}

impl <V: Vertex> Mesh<V> {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, config: MeshConfiguration<V>) -> Result<Self> {

        let buffer_create_info = BufferCreateInfo{
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };

        let vertex_buffer = Buffer::from_iter(allocator.clone(), buffer_create_info, allocation_info, config.vertices)?;

        let mut index_buffer = None;
        
        if let Some(indices) = config.indices {
            let buffer_create_info = BufferCreateInfo{
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            };

            let allocation_info = AllocationCreateInfo{
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            };

            let buf = Buffer::from_iter(allocator.clone(), buffer_create_info, allocation_info, indices)?;
            index_buffer = Some(buf);
        }
        

        let mesh = Mesh{
            vertex_buffer,
            index_buffer,
        };

        Ok(mesh)
    }
    
    pub fn draw(&self, cmd_buf: &mut PrimaryCommandBuffer, instance_count: u32, first_instance: u32) -> Result<()> {

        cmd_buf.bind_vertex_buffers(0, self.vertex_buffer.clone())?;
        
        if let Some(index_buffer) = &self.index_buffer {
            cmd_buf.bind_index_buffer(index_buffer.clone())?;
            unsafe { cmd_buf.draw_indexed(index_buffer.len() as u32, instance_count, 0, 0, first_instance) }?;
        } else {
            unsafe { cmd_buf.draw(self.vertex_buffer.len() as u32, instance_count, 0, first_instance) }?;
        }
        
        Ok(())
    }
}

