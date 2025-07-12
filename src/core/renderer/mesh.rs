use std::cmp::Ordering;
use crate::core::{Engine, PrimaryCommandBuffer};
use anyhow::Result;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::Vertex;

pub struct MeshConfiguration<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Option<Vec<u32>>,
}



#[derive(Clone)]
pub struct Mesh<V: Vertex> {
    vertex_buffer: Subbuffer<[V]>,
    index_buffer: Option<Subbuffer<[u32]>>,
    resource_id: u64,
}

impl<V: Vertex> PartialEq<Self> for Mesh<V> {
    fn eq(&self, other: &Self) -> bool {
        self.resource_id == other.resource_id
    }
}

impl <V: Vertex> Eq for Mesh<V> {}

impl<V: Vertex> PartialOrd<Self> for Mesh<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.resource_id.partial_cmp(&other.resource_id)
    }
}
impl<V: Vertex> Ord for Mesh<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resource_id.cmp(&other.resource_id)
    }
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
        
        let resource_id = Engine::next_resource_id();

        let mesh = Mesh{
            vertex_buffer,
            index_buffer,
            resource_id
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
    
    pub fn get_resource_id(&self) -> u64 {
        self.resource_id
    }
}

