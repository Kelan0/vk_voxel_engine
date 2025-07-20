use crate::core::{CommandBuffer, Engine, GraphicsManager, PrimaryCommandBuffer};
use anyhow::Result;
use std::cmp::Ordering;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::CopyBufferInfo;
use vulkano::DeviceSize;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
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
        Some(self.cmp(other))
    }
}
impl<V: Vertex> Ord for Mesh<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resource_id.cmp(&other.resource_id)
    }
}


impl <V: Vertex> Mesh<V> {
    pub fn new(allocator: Arc<dyn MemoryAllocator>, config: MeshConfiguration<V>) -> Result<Self> {

        let vertex_buffer = Self::create_and_upload_vertex_buffer(allocator.clone(), config.vertices)?;
        let index_buffer = Self::create_and_upload_index_buffer(allocator.clone(), config.indices)?;
        
        let resource_id = Engine::next_resource_id();

        let mesh = Mesh{
            vertex_buffer,
            index_buffer,
            resource_id
        };

        Ok(mesh)
    }
    pub fn new_staged<L>(allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer<L>, config: MeshConfiguration<V>) -> Result<Self> {

        let vertex_buffer = Self::create_and_upload_vertex_buffer_staged(allocator.clone(), cmd_buf, config.vertices)?;
        let index_buffer = Self::create_and_upload_index_buffer_staged(allocator.clone(), cmd_buf, config.indices)?;
        
        let resource_id = Engine::next_resource_id();

        let mesh = Mesh{
            vertex_buffer,
            index_buffer,
            resource_id
        };

        Ok(mesh)
    }
    
    fn create_vertex_buffer(allocator: Arc<dyn MemoryAllocator>, vertex_count: usize, host_write: bool) -> Result<Subbuffer<[V]>> {

        let buffer_create_info = BufferCreateInfo{
            usage: if host_write { 
                BufferUsage::VERTEX_BUFFER 
            } else {
                BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST
            },
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: if host_write { 
                MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE 
            } else { 
                MemoryTypeFilter::PREFER_DEVICE 
            }, 
            ..Default::default()
        };

        let vertex_buffer = Buffer::new_slice::<V>(allocator.clone(), buffer_create_info, allocation_info, vertex_count as DeviceSize)?;

        Ok(vertex_buffer)
    }


    fn create_index_buffer(allocator: Arc<dyn MemoryAllocator>, index_count: usize, host_write: bool) -> Result<Subbuffer<[u32]>> {
        
        let buffer_create_info = BufferCreateInfo{
            usage: if host_write {
                BufferUsage::INDEX_BUFFER
            } else {
                BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST
            },
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            memory_type_filter: if host_write {
                MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
            } else {
                MemoryTypeFilter::PREFER_DEVICE
            },
            ..Default::default()
        };

        // let buf = Buffer::from_iter(allocator.clone(), buffer_create_info, allocation_info, indices)?;
        let index_buffer = Buffer::new_slice::<u32>(allocator.clone(), buffer_create_info, allocation_info, index_count as DeviceSize)?;
        Ok(index_buffer)
    }
    
    fn create_and_upload_vertex_buffer(allocator: Arc<dyn MemoryAllocator>, vertices: Vec<V>) -> Result<Subbuffer<[V]>> {
        let vertex_buffer = Self::create_vertex_buffer(allocator, vertices.len(), true)?;
        GraphicsManager::upload_buffer_data_iter(&vertex_buffer, vertices)?;
        Ok(vertex_buffer)
    }
    
    fn create_and_upload_vertex_buffer_staged<L>(allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer<L>, vertices: Vec<V>) -> Result<Subbuffer<[V]>> {
        let vertex_buffer = Self::create_vertex_buffer(allocator.clone(), vertices.len(), false)?;
        let staging_buffer = GraphicsManager::create_staging_subbuffer::<V>(allocator, vertices.len() as DeviceSize)?;
        GraphicsManager::upload_buffer_data_iter(&staging_buffer, vertices)?;
        cmd_buf.copy_buffer(CopyBufferInfo::buffers(staging_buffer, vertex_buffer.clone()))?;
        Ok(vertex_buffer)
    }
    
    fn create_and_upload_index_buffer(allocator: Arc<dyn MemoryAllocator>, indices: Option<Vec<u32>>) -> Result<Option<Subbuffer<[u32]>>> {

        let index_buffer = match indices {
            Some(indices) => {
                let index_buffer = Self::create_index_buffer(allocator.clone(), indices.len(), true)?;
                GraphicsManager::upload_buffer_data_iter(&index_buffer, indices)?;
                Some(index_buffer)
            }
            None => None
        };
        
        Ok(index_buffer)
    }

    fn create_and_upload_index_buffer_staged<L>(allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer<L>, indices: Option<Vec<u32>>) -> Result<Option<Subbuffer<[u32]>>> {
        let index_buffer = match indices {
            Some(indices) => {
                let index_buffer = Self::create_index_buffer(allocator.clone(), indices.len(), false)?;
                let staging_buffer = GraphicsManager::create_staging_subbuffer::<u32>(allocator, indices.len() as DeviceSize)?;
                GraphicsManager::upload_buffer_data_iter(&staging_buffer, indices)?;
                cmd_buf.copy_buffer(CopyBufferInfo::buffers(staging_buffer, index_buffer.clone()))?;
                Some(index_buffer)
            }
            None => None
        };

        Ok(index_buffer)
    }
    
    pub fn upload(&mut self, allocator: Arc<dyn MemoryAllocator>, config: MeshConfiguration<V>) -> Result<()> {
        self.vertex_buffer = Self::create_and_upload_vertex_buffer(allocator.clone(), config.vertices)?;
        self.index_buffer = Self::create_and_upload_index_buffer(allocator.clone(), config.indices)?;

        Ok(())
    }
    
    pub fn upload_staged<L>(&mut self, allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer<L>, config: MeshConfiguration<V>) -> Result<()>{
        self.vertex_buffer = Self::create_and_upload_vertex_buffer_staged(allocator.clone(), cmd_buf, config.vertices)?;
        self.index_buffer = Self::create_and_upload_index_buffer_staged(allocator.clone(), cmd_buf, config.indices)?;

        Ok(())
    }

    fn get_raw_bytes<T>(data: &T) -> &[u8] {
        let ptr = data as *const T as *const u8;
        let len = size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    fn get_raw_bytes_slice<T>(data: &[T]) -> &[u8] {
        let ptr = data as *const [T] as *const u8;
        let len = size_of_val(data);
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    pub fn resource_id(&self) -> u64 {
        self.resource_id
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

