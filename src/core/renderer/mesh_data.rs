use crate::core::{set_vulkan_debug_name, util, AxisAlignedBoundingBox, CommandBuffer, GraphicsManager, Mesh, MeshBufferOption, MeshConfiguration, MeshPrimitiveType, Transform};
use anyhow::Result;
use ash::vk::DeviceSize;
use glam::{Affine3A, DVec3, IVec3, Mat4, Vec3};
use std::fmt::{Debug, Formatter};
use std::ops::RangeBounds;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::memory::allocator::{align_up, MemoryAllocator};
use vulkano::memory::DeviceAlignment;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::MeshBufferOption::AllocateNew;

#[derive(Clone, PartialEq)]
pub struct MeshData<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
    primitive_type: MeshPrimitiveType,
    transform_stack: Vec<Transform>,
    current_transform: Transform,
    has_indices: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisDirection {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
}

impl AxisDirection {
    pub fn from_index(index: u32) -> Option<AxisDirection> {
        match index {
            0 => Some(AxisDirection::NegX),
            1 => Some(AxisDirection::PosX),
            2 => Some(AxisDirection::NegY),
            3 => Some(AxisDirection::PosY),
            4 => Some(AxisDirection::NegZ),
            5 => Some(AxisDirection::PosZ),
            _ => None
        }
    }
    pub fn index(&self) -> u32 {
        match *self {
            AxisDirection::NegX => 0,
            AxisDirection::PosX => 1,
            AxisDirection::NegY => 2,
            AxisDirection::PosY => 3,
            AxisDirection::NegZ => 4,
            AxisDirection::PosZ => 5,
        }
    }

    pub fn vec(&self) -> Vec3 {
        match *self {
            AxisDirection::NegX => Vec3::NEG_X,
            AxisDirection::PosX => Vec3::X,
            AxisDirection::NegY => Vec3::NEG_Y,
            AxisDirection::PosY => Vec3::Y,
            AxisDirection::NegZ => Vec3::NEG_Z,
            AxisDirection::PosZ => Vec3::Z,
        }
    }

    pub fn dvec(&self) -> DVec3 {
        match *self {
            AxisDirection::NegX => DVec3::NEG_X,
            AxisDirection::PosX => DVec3::X,
            AxisDirection::NegY => DVec3::NEG_Y,
            AxisDirection::PosY => DVec3::Y,
            AxisDirection::NegZ => DVec3::NEG_Z,
            AxisDirection::PosZ => DVec3::Z,
        }
    }

    pub fn ivec(&self) -> IVec3 {
        match *self {
            AxisDirection::NegX => IVec3::NEG_X,
            AxisDirection::PosX => IVec3::X,
            AxisDirection::NegY => IVec3::NEG_Y,
            AxisDirection::PosY => IVec3::Y,
            AxisDirection::NegZ => IVec3::NEG_Z,
            AxisDirection::PosZ => IVec3::Z,
        }
    }

    pub fn opposite(&self) -> AxisDirection {
        match *self {
            AxisDirection::NegX => AxisDirection::PosX,
            AxisDirection::PosX => AxisDirection::NegX,
            AxisDirection::NegY => AxisDirection::PosY,
            AxisDirection::PosY => AxisDirection::NegY,
            AxisDirection::NegZ => AxisDirection::PosZ,
            AxisDirection::PosZ => AxisDirection::NegZ,
        }

    }

    pub fn name(&self) -> &str {
        match *self {
            AxisDirection::NegX => "NegX",
            AxisDirection::PosX => "PosX",
            AxisDirection::NegY => "NegY",
            AxisDirection::PosY => "PosY",
            AxisDirection::NegZ => "NegZ",
            AxisDirection::PosZ => "PosZ",
        }
    }
}


#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MeshDataConfig {
    pub primitive_type: MeshPrimitiveType,
    pub has_indices: bool
}

impl MeshDataConfig {
    pub fn new(primitive_type: MeshPrimitiveType) -> Self {
        MeshDataConfig {
            primitive_type,
            has_indices: true,
        }
    }

    pub fn without_indices(mut self) -> Self {
        self.has_indices = false;
        self
    }
}

impl <V: Vertex> MeshData<V> {

    pub fn new(config: MeshDataConfig) -> Self {
        MeshData{
            primitive_type: config.primitive_type,
            has_indices: config.has_indices,
            ..Default::default()
        }
    }

    pub fn reserve_vertices(&mut self, additional_vertex_cound: usize) {
        self.vertices.reserve(additional_vertex_cound)
    }

    pub fn reserve_indices(&mut self, additional_index_cound: usize) {
        self.indices.reserve(additional_index_cound)
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn primitive_type(&self) -> MeshPrimitiveType {
        self.primitive_type
    }

    pub fn calculate_aabb(&self) -> AxisAlignedBoundingBox
    where V: VertexHasPosition<f32> {
        let mut pos_min = [f32::MAX, f32::MAX, f32::MAX];
        let mut pos_max = [f32::MIN, f32::MIN, f32::MIN];

        for vertex in &self.vertices {
            let p = vertex.position();

            pos_min[0] = f32::min(pos_min[0], p[0]);
            pos_min[1] = f32::min(pos_min[1], p[1]);
            pos_min[2] = f32::min(pos_min[2], p[2]);

            pos_max[0] = f32::max(pos_max[0], p[0]);
            pos_max[1] = f32::max(pos_max[1], p[1]);
            pos_max[2] = f32::max(pos_max[2], p[2]);
        }

        AxisAlignedBoundingBox::new(
            DVec3::new(
                pos_min[0] as f64,
                pos_min[1] as f64,
                pos_min[2] as f64
            ),
            DVec3::new(
                pos_max[0] as f64,
                pos_max[1] as f64,
                pos_max[2] as f64
            )
        )
    }

    pub fn add_vertex(&mut self, vertex: V) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        index
    }

    pub fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        match self.primitive_type {
            MeshPrimitiveType::TriangleList => {
                self.create_triangle_primitive(i0, i1, i2)
            },
            MeshPrimitiveType::LineList => {
                let i = self.create_line_primitive(i0, i1);
                let _ = self.create_line_primitive(i1, i2);
                let _ = self.create_line_primitive(i2, i0);
                i
            }
        }

    }

    pub fn add_quad(&mut self, i0: u32, i1: u32, i2: u32, i3: u32) -> u32 {
        match self.primitive_type {
            MeshPrimitiveType::TriangleList => {
                let i = self.create_triangle_primitive(i0, i1, i2);
                let _ = self.create_triangle_primitive(i0, i2, i3);
                i
            }
            MeshPrimitiveType::LineList => {
                let i = self.create_line_primitive(i0, i1);
                let _ = self.create_line_primitive(i1, i2);
                let _ = self.create_line_primitive(i2, i3);
                let _ = self.create_line_primitive(i3, i0);
                i
            }
        }
    }

    pub fn add_line(&mut self, i0: u32, i1: u32) -> u32 {
        assert_eq!(self.primitive_type, MeshPrimitiveType::LineList);
        self.create_line_primitive(i0, i1)
    }
    
    pub fn create_quad(&mut self, v00: V, v01: V, v11: V, v10: V) -> (u32, u32) {
        if self.has_indices {

            let vert_idx = self.vertices.len() as u32;
            let i00 = self.add_vertex(v00);
            let i01 = self.add_vertex(v01);
            let i11 = self.add_vertex(v11);
            let i10 = self.add_vertex(v10);
            let index_idx = self.add_quad(i00, i01, i11, i10);
            (vert_idx, index_idx)

        } else {

            let vert_idx = self.vertices.len() as u32;
            let i00 = self.add_vertex(v00);
            let i01 = self.add_vertex(v01);
            let i11 = self.add_vertex(v11);
            let i10 = self.add_vertex(v10);
            let index_idx = self.add_quad(i00, i01, i11, i10);
            (vert_idx, index_idx)
        }
    }

    fn create_triangle_primitive(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        debug_assert_eq!(self.primitive_type, MeshPrimitiveType::TriangleList);
        let index = self.indices.len() as u32;
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
        index
    }

    fn create_line_primitive(&mut self, i0: u32, i1: u32) -> u32 {
        debug_assert_eq!(self.primitive_type, MeshPrimitiveType::LineList);
        let index = self.indices.len() as u32;
        self.indices.push(i0);
        self.indices.push(i1);
        index
    }

    pub fn push_transform(&mut self) -> u32 {
        self.transform_stack.push(self.current_transform.clone());
        self.vertices.len() as u32
    }

    pub fn pop_transform(&mut self) -> u32 {
        debug_assert!(!self.transform_stack.is_empty(), "MeshData::pop_transform(): Stack underflow");
        let transform = self.transform_stack.pop().unwrap();
        self.current_transform = transform;
        self.vertices.len() as u32
    }

    pub fn pop_transform_apply(&mut self, start_index: u32) -> u32
    where V: VertexHasPosition<f32> {
        self.apply_transform(start_index..);
        self.pop_transform()
    }

    pub fn apply_transform<R>(&mut self, range: R)
    where V: VertexHasPosition<f32>,
          R: RangeBounds<u32> {

        let (start, end) = util::get_range(range, self.vertices.len() as u32);

        for i in start..end {
            self.vertices[i as usize].transform_affine(self.current_transform.affine);
        }
    }

    pub fn transform(&mut self) -> &mut Transform {
        &mut self.current_transform
    }
    
    pub fn build_mesh(self, allocator: Arc<dyn MemoryAllocator>) -> Result<Mesh<V>> {
        let mesh = Mesh::new(allocator, MeshConfiguration {
            primitive_type: self.primitive_type,
            vertices: self.vertices,
            indices: Some(self.indices),
            vertex_buffer: AllocateNew,
            index_buffer: AllocateNew,
        })?;
        
        Ok(mesh)
    }

    pub fn build_mesh_ref(&self, allocator: Arc<dyn MemoryAllocator>) -> Result<Mesh<V>>
    where V: Clone {
        let mesh = Mesh::new(allocator, MeshConfiguration {
            primitive_type: self.primitive_type,
            vertices: self.vertices.to_vec(),
            indices: Some(self.indices.to_vec()),
            vertex_buffer: AllocateNew,
            index_buffer: AllocateNew,
        })?;

        Ok(mesh)
    }
    
    pub fn build_mesh_staged(self, allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer, staging_buffer: &Subbuffer<[u8]>) -> Result<Mesh<V>> {
        let mesh = Mesh::new_staged(allocator.clone(), cmd_buf, staging_buffer, MeshConfiguration {
            primitive_type: self.primitive_type,
            vertices: self.vertices,
            indices: Some(self.indices),
            vertex_buffer: AllocateNew,
            index_buffer: AllocateNew,
        })?;
        
        Ok(mesh)
    }

    pub fn build_mesh_staged_ref(&self, allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer, staging_buffer: &Subbuffer<[u8]>) -> Result<Mesh<V>>
    where V: Clone {
        let mesh = Mesh::new_staged(allocator.clone(), cmd_buf, staging_buffer, MeshConfiguration {
            primitive_type: self.primitive_type,
            vertices: self.vertices.to_vec(),
            indices: Some(self.indices.to_vec()),
            vertex_buffer: AllocateNew,
            index_buffer: AllocateNew,
        })?;

        Ok(mesh)
    }

    pub fn upload_vertex_data(&self, buffer: &Subbuffer<[u8]>) -> Result<()> {
        debug_assert!(buffer.size() >= self.get_required_vertex_buffer_size());

        GraphicsManager::upload_buffer_data_bytes_iter_ref(&buffer, &self.vertices)?;
        Ok(())
    }

    pub fn upload_index_data(&self, buffer: &Subbuffer<[u8]>) -> Result<bool> {
        if self.indices.is_empty() {
            return Ok(false);
        }

        debug_assert!(buffer.size() >= self.get_required_index_buffer_size());
        GraphicsManager::upload_buffer_data_bytes_iter_ref(&buffer, &self.indices)?;
        Ok(true)
    }

    pub fn upload_mesh_data(&self, buffer: &Subbuffer<[u8]>) -> Result<()>{
        self.upload_vertex_data(buffer)?;

        if !self.indices.is_empty() {
            let offset = self.get_required_vertex_buffer_size();
            let buffer = buffer.clone().slice(offset..);
            self.upload_index_data(&buffer)?;
        }

        Ok(())
    }
    
    pub fn get_required_vertex_buffer_size(&self) -> DeviceSize {
        Self::calc_required_vertex_buffer_size(self.vertices.len())
    }
    
    pub fn get_required_index_buffer_size(&self) -> DeviceSize {
        Self::calc_required_index_buffer_size(self.indices.len())
    }
    
    pub fn get_required_staging_buffer_size(&self) -> DeviceSize {
        self.get_required_vertex_buffer_size() + self.get_required_index_buffer_size()
    }
    
    pub fn create_staging_buffer(&self, allocator: Arc<dyn MemoryAllocator>) -> Result<Subbuffer<[u8]>>{
        let required_len = self.get_required_staging_buffer_size();
        let buf = GraphicsManager::create_staging_subbuffer(allocator, required_len)?;
        set_vulkan_debug_name(buf.buffer(), Some("MeshData-StagingBuffer"))?;
        Ok(buf)
    }
    
    pub fn calc_required_vertex_buffer_size(vertex_count: usize) -> DeviceSize {
        let size = (vertex_count * size_of::<V>()) as DeviceSize;
        
        if let Some(alignment) = DeviceAlignment::new(align_of::<V>() as DeviceSize) {
            align_up(size, alignment)
        } else {
            size
        }
    }
    
    pub fn calc_required_index_buffer_size(index_count: usize) -> DeviceSize {
        let size = (index_count * size_of::<u32>()) as DeviceSize;
        
        if let Some(alignment) = DeviceAlignment::new(align_of::<u32>() as DeviceSize) {
            align_up(size, alignment)
        } else {
            size
        }
    }
}



impl <V: Vertex + Default> MeshData<V> {
    pub fn vertex(&mut self) -> VertexBuilder<V> {
        VertexBuilder::new(self)
    }
}



impl <V: Vertex + Default> MeshData<V> {

    pub fn texture_quad<T>(&mut self, start_index: u32, pos_btm_left: [T; 2], pos_btm_right: [T; 2], pos_top_right: [T; 2], pos_top_left: [T; 2])
    where V: VertexHasTexture<T>,
          T: Copy {

        self.vertices[start_index as usize].set_texture(pos_btm_left);
        self.vertices[(start_index + 1) as usize].set_texture(pos_btm_right);
        self.vertices[(start_index + 2) as usize].set_texture(pos_top_right);
        self.vertices[(start_index + 3) as usize].set_texture(pos_top_left);
    }

    pub fn colour_vertices<R, T>(&mut self, range: R, colour: [T; 4])
    where V: VertexHasColour<T>,
          R: RangeBounds<u32>,
          T: Copy {

        let (start, end) = util::get_range(range, self.vertices.len() as u32);

        for i in start as usize ..end as usize {
            self.vertices[i].set_colour(colour);
        }
    }

    #[allow(clippy::too_many_arguments)] // grr >:(
    pub fn create_quad_face(&mut self, center: [f32; 3], pos_btm_left: [f32; 2], pos_btm_right: [f32; 2], pos_top_right: [f32; 2], pos_top_left: [f32; 2], left: [f32; 3], up: [f32; 3], offset: f32) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {
        let up = Vec3::from_slice(&up);
        let left = Vec3::from_slice(&left);
        let normal = Vec3::cross(up, left);
        let center = Vec3::from_slice(&center);
        
        let pos_btm_left = center + left * pos_btm_left[0] + up * pos_btm_left[1] + normal * offset;
        let pos_btm_right = center + left * pos_btm_right[0] + up * pos_btm_right[1] + normal * offset;
        let pos_top_right = center + left * pos_top_right[0] + up * pos_top_right[1] + normal * offset;
        let pos_top_left = center + left * pos_top_left[0] + up * pos_top_left[1] + normal * offset;
        
        let v00 = self.vertex().pos(pos_btm_left.into()).normal(normal.into()).get();
        let v01 = self.vertex().pos(pos_btm_right.into()).normal(normal.into()).get();
        let v11 = self.vertex().pos(pos_top_right.into()).normal(normal.into()).get();
        let v10 = self.vertex().pos(pos_top_left.into()).normal(normal.into()).get();
        
         self.create_quad(v00, v01, v11, v10)
    }
    
    pub fn create_box_face(&mut self, direction: AxisDirection, center: [f32; 3], half_size: [f32; 2], offset: f32) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {

        let pos_min = [-half_size[0], -half_size[1]];
        let pos_max = half_size;
        
        match direction {
            AxisDirection::NegX => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::PosX => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::NegY => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], offset),
            AxisDirection::PosY => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], offset),
            AxisDirection::NegZ => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::PosZ => self.create_quad_face(center, [pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], offset),
        }
    }
    
    pub fn create_box_face_textured<T>(&mut self, direction: AxisDirection, center: [f32; 3], half_size: [f32; 2], offset: f32, tex_min: [T; 2], tex_max: [T; 2]) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> + VertexHasTexture<T>,
          T: Copy {

        const FLIP_Y: bool = true;
        let (vert_idx, index_idx) = self.create_box_face(direction, center, half_size, offset);
        if FLIP_Y {
            self.texture_quad(vert_idx, [tex_min[0], tex_max[1]], [tex_max[0], tex_max[1]], [tex_max[0], tex_min[1]], [tex_min[0], tex_min[1]]);
        } else {
            self.texture_quad(vert_idx, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        }
        (vert_idx, index_idx)
    }

    pub fn create_cuboid(&mut self, center: [f32; 3], half_size: [f32; 3]) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {

        let idx =
            self.create_box_face(AxisDirection::NegX, center, [half_size[1], half_size[2]], half_size[0]);
        self.create_box_face(AxisDirection::PosX, center, [half_size[1], half_size[2]], half_size[0]);
        self.create_box_face(AxisDirection::NegY, center, [half_size[0], half_size[2]], half_size[1]);
        self.create_box_face(AxisDirection::PosY, center, [half_size[0], half_size[2]], half_size[1]);
        self.create_box_face(AxisDirection::NegZ, center, [half_size[0], half_size[1]], half_size[2]);
        self.create_box_face(AxisDirection::PosZ, center, [half_size[0], half_size[1]], half_size[2]);
        idx
    }

    pub fn create_cuboid_textured<T>(&mut self, center: [f32; 3], half_size: [f32; 3], tex_min: [T; 2], tex_max: [T; 2]) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> + VertexHasTexture<T>,
          T: Copy {
        
        let (vert_idx, index_idx) = self.create_cuboid(center, half_size);
        
        self.texture_quad(vert_idx, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(vert_idx + 4, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(vert_idx + 8, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(vert_idx + 12, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(vert_idx + 16, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(vert_idx + 20, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);

        (vert_idx, index_idx)
    }

    pub fn create_plane(&mut self, center: [f32; 3], u: [f32; 3], v: [f32; 3], extent: [f32; 2], cells: [u32; 2]) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {
        let u = Vec3::from_slice(&u);
        let v = Vec3::from_slice(&v);
        let normal = Vec3::cross(v, u);
        let n = normal.into();
        let center = Vec3::from_slice(&center);

        let vertex_idx = self.vertices.len() as u32;
        let index_idx = self.indices.len() as u32;

        let cells_u = cells[0];
        let cells_v = cells[1];
        let verts_u = cells_u + 1;
        let verts_v = cells_v + 1;

        let delta_u = extent[0] / cells_u as f32;
        let delta_v = extent[1] / cells_v as f32;

        let hu = cells_u as f32 * 0.5;
        let hv = cells_v as f32 * 0.5;

        for i in 0 .. verts_u {
            let offset_u = delta_u * (i as f32 - hu);
            let pos_u = u * offset_u;

            for j in 0 .. verts_v {
                let offset_v = delta_v * (j as f32 - hv);
                let pos_v = v * offset_v;

                let p = center + pos_u + pos_v;
                self.vertex().pos(p.into()).normal(n).add();
            }
        }


        for i in 0 ..cells_u {
            let a = vertex_idx + (i * verts_v);
            let b = vertex_idx + ((i + 1) * verts_v);

            for j in 0 ..cells_v {
                let i00 = a + j;
                let i10 = b + j;
                let i01 = a + (j + 1);
                let i11 = b + (j + 1);

                self.add_quad(i00, i01, i11, i10);
            }
        }

        (vertex_idx, index_idx)
    }

    pub fn create_lines_grid(&mut self, center: [f32; 3], u: [f32; 3], v: [f32; 3], extent: [f32; 2], cells: [u32; 2]) -> (u32, u32)
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {
        debug_assert_eq!(self.primitive_type, MeshPrimitiveType::LineList);

        let u = Vec3::from_slice(&u);
        let v = Vec3::from_slice(&v);
        let normal = Vec3::cross(v, u);
        let n = normal.into();
        let center = Vec3::from_slice(&center);

        let vertex_idx = self.vertices.len() as u32;
        let index_idx = self.indices.len() as u32;

        let cells_u = cells[0];
        let cells_v = cells[1];
        let verts_u = cells_u + 1;
        let verts_v = cells_v + 1;

        let delta_u = extent[0] / cells_u as f32;
        let delta_v = extent[1] / cells_v as f32;

        let hu = cells_u as f32 * 0.5;
        let hv = cells_v as f32 * 0.5;

        let pos_u0 = u * delta_u * -hu;
        let pos_u1 = u * delta_u * hu;
        let pos_v0 = v * delta_v * -hv;
        let pos_v1 = v * delta_v * hv;

        for i in 0 .. verts_u {
            let offset_u = delta_u * (i as f32 - hu);
            let pos_u = u * offset_u;

            // Vertical line
            let p0 = center + pos_v0 + pos_u;
            let p1 = center + pos_v1 + pos_u;
            let i0 = self.vertex().pos(p0.into()).normal(n).add();
            let i1 = self.vertex().pos(p1.into()).normal(n).add();
            self.add_line(i0, i1);
        }

        for j in 0 .. verts_v {
            let offset_v = delta_v * (j as f32 - hv);
            let pos_v = v * offset_v;

            // Horizontal line
            let p0 = center + pos_u0 + pos_v;
            let p1 = center + pos_u1 + pos_v;
            let i0 = self.vertex().pos(p0.into()).normal(n).add();
            let i1 = self.vertex().pos(p1.into()).normal(n).add();
            self.add_line(i0, i1);
        }

        (vertex_idx, index_idx)
    }
}





pub trait VertexHasPosition<T>: Default {
    fn position(&self) -> &[T; 3];
    fn set_position(&mut self, pos: [T; 3]);
    fn transform_mat4(&mut self, transform: Mat4);
    fn transform_affine(&mut self, transform: Affine3A);
}

pub trait VertexHasNormal<T>: Default {
    fn normal(&self) -> &[T; 3];
    fn set_normal(&mut self, normal: [T; 3]);
}

pub trait VertexHasColour<T>: Default {
    fn colour(&self) -> &[T; 4];
    fn set_colour(&mut self, colour: [T; 4]);
}

pub trait VertexHasTexture<T>: Default {
    fn texture(&self) -> &[T; 2];
    fn set_texture(&mut self, colour: [T; 2]);
}



pub struct VertexBuilder<'a, V: Vertex> {
    mesh_data: &'a mut MeshData<V>,
    vertex: V
}

impl <'a, V: Vertex + Default> VertexBuilder<'a, V> {
    pub fn new(mesh_data: &'a mut MeshData<V>) -> Self {
        VertexBuilder{
            mesh_data,
            vertex: Default::default()
        }
    }

    pub fn get(self) -> V {
        self.vertex
    }

    pub fn add(self) -> u32 {
        self.mesh_data.add_vertex(self.vertex)
    }
}

impl <'a, V: Vertex + Default + Clone> VertexBuilder<'a, V> {
    pub fn get_c(&self) -> V {
        self.vertex.clone()
    }

    pub fn add_c(&mut self) -> u32 {
        self.mesh_data.add_vertex(self.vertex.clone())
    }
}


impl <'a, V> VertexBuilder<'a, V>
where V: Vertex {
    pub fn pos<T>(mut self, pos: [T; 3]) -> Self
    where 
        V: VertexHasPosition<T> {
        self.vertex.set_position(pos);
        self
    }
}

impl <'a, V> VertexBuilder<'a, V>
where V: Vertex {
    pub fn normal<T>(mut self, norm: [T; 3]) -> Self
    where
        V: VertexHasNormal<T> {
        self.vertex.set_normal(norm);
        self
    }
}

impl <'a, V> VertexBuilder<'a, V>
where V: Vertex {
    pub fn colour<T>(mut self, colour: [T; 4]) -> Self
    where
        V: VertexHasColour<T> {
        self.vertex.set_colour(colour);
        self
    }
}

impl <'a, V> VertexBuilder<'a, V>
where V: Vertex {
    pub fn texture<T>(mut self, texture: [T; 2]) -> Self
    where
        V: VertexHasTexture<T> {
        self.vertex.set_texture(texture);
        self
    }
}


impl <V: Vertex> Default for MeshData<V> {
    fn default() -> Self {
        Self{
            primitive_type: MeshPrimitiveType::TriangleList,
            transform_stack: vec![],
            vertices: vec![],
            indices: vec![],
            current_transform: Transform::default(),
            has_indices: true,
        }
    }
}

impl<V> Debug for MeshData<V>
where V: Vertex + Debug {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeshData{{\n\tprimitive_type:{:?},\n\tvertices:[\n", self.primitive_type)?;
        for (i, vertex) in self.vertices.iter().enumerate() {
            write!(f, "\t\t[{i}] = {vertex:?}\n")?;
        }
        write!(f, "\t],\n\tindices:[\n\t\t")?;
        for (i, index) in self.indices.iter().enumerate() {
            if *index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{index}")?;
        }
        write!(f, "\n\t]\n }}")?;
        Ok(())
    }
}