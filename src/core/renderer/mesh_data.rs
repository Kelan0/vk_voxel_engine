use crate::core::{CommandBuffer, Mesh, MeshConfiguration};
use anyhow::Result;
use glam::Vec3;
use std::sync::Arc;
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(Clone, PartialEq)]
pub struct MeshData<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
}

pub enum AxisDirection {
    NegX,
    PosX,
    NegY,
    PosY,
    NegZ,
    PosZ,
}

impl <V: Vertex> MeshData<V> {

    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_vertex(&mut self, vertex: V) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        index
    }

    pub fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        self.create_triangle_primitive(i0, i1, i2)
    }

    pub fn add_quad(&mut self, i0: u32, i1: u32, i2: u32, i3: u32) -> u32 {
        let i = self.create_triangle_primitive(i0, i1, i2);
        let _ = self.create_triangle_primitive(i0, i2, i3);
        i
    }
    
    pub fn create_quad(&mut self, v00: V, v01: V, v11: V, v10: V) -> (u32, u32) {
        let vert_idx = self.vertices.len() as u32;
        let i00 = self.add_vertex(v00);
        let i01 = self.add_vertex(v01);
        let i11 = self.add_vertex(v11);
        let i10 = self.add_vertex(v10);
        let index_idx = self.add_quad(i00, i01, i11, i10);
        (vert_idx, index_idx)
    }

    fn create_triangle_primitive(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        let index = self.indices.len() as u32;
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
        index
    }
    
    pub fn build_mesh(self, allocator: Arc<dyn MemoryAllocator>) -> Result<Mesh<V>> {
        let mesh = Mesh::new(allocator, MeshConfiguration {
            vertices: self.vertices,
            indices: Some(self.indices),
        })?;
        
        Ok(mesh)
    }
    
    pub fn build_mesh_staged<L>(self, allocator: Arc<dyn MemoryAllocator>, cmd_buf: &mut CommandBuffer<L>) -> Result<Mesh<V>> {
        let mesh = Mesh::new_staged(allocator.clone(), cmd_buf, MeshConfiguration {
            vertices: self.vertices,
            indices: Some(self.indices),
        })?;
        
        Ok(mesh)
    }
}



impl <V: Vertex + Default> MeshData<V> {
    pub fn vertex(&mut self) -> VertexBuilder<V> {
        VertexBuilder::new(self)
    }
}



impl <V: Vertex + Default> MeshData<V> {

    pub fn texture_quad(&mut self, start_index: u32, pos_btm_left: [f32; 2], pos_btm_right: [f32; 2], pos_top_right: [f32; 2], pos_top_left: [f32; 2])
    where V: VertexHasTexture<f32> {
        self.vertices[start_index as usize].set_texture(pos_btm_left);
        self.vertices[(start_index + 1) as usize].set_texture(pos_btm_right);
        self.vertices[(start_index + 2) as usize].set_texture(pos_top_right);
        self.vertices[(start_index + 3) as usize].set_texture(pos_top_left);
    }
    
    #[allow(clippy::too_many_arguments)] // grr >:(
    pub fn create_quad_face(&mut self, pos_btm_left: [f32; 2], pos_btm_right: [f32; 2], pos_top_right: [f32; 2], pos_top_left: [f32; 2], left: [f32; 3], up: [f32; 3], offset: f32) -> u32
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {
        let up = Vec3::from_slice(&up);
        let left = Vec3::from_slice(&left);
        let normal = Vec3::cross(up, left);
        
        let pos_btm_left = left * pos_btm_left[0] + up * pos_btm_left[1] + normal * offset;
        let pos_btm_right = left * pos_btm_right[0] + up * pos_btm_right[1] + normal * offset;
        let pos_top_right = left * pos_top_right[0] + up * pos_top_right[1] + normal * offset;
        let pos_top_left = left * pos_top_left[0] + up * pos_top_left[1] + normal * offset;
        
        let v00 = self.vertex().pos(pos_btm_left.into()).normal(normal.into()).get();
        let v01 = self.vertex().pos(pos_btm_right.into()).normal(normal.into()).get();
        let v11 = self.vertex().pos(pos_top_right.into()).normal(normal.into()).get();
        let v10 = self.vertex().pos(pos_top_left.into()).normal(normal.into()).get();
        
        let (idx, _) = self.create_quad(v00, v01, v11, v10);
        idx
    }
    
    pub fn create_box_face(&mut self, direction: AxisDirection, pos_min: [f32; 2], pos_max: [f32; 2], offset: f32) -> u32
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {

        match direction {
            AxisDirection::NegX => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::PosX => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::NegY => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], offset),
            AxisDirection::PosY => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], offset),
            AxisDirection::NegZ => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], offset),
            AxisDirection::PosZ => self.create_quad_face([pos_min[0], pos_min[1]], [pos_max[0], pos_min[1]], [pos_max[0], pos_max[1]], [pos_min[0], pos_max[1]], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], offset),
        }
    }
    
    pub fn create_box_face_textured(&mut self, direction: AxisDirection, pos_min: [f32; 2], pos_max: [f32; 2], offset: f32, tex_min: [f32; 2], tex_max: [f32; 2]) -> u32
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> + VertexHasTexture<f32> {
        const FLIP_Y: bool = true;
        let index = self.create_box_face(direction, pos_min, pos_max, offset);
        if FLIP_Y {
            self.texture_quad(index, [tex_min[0], tex_max[1]], [tex_max[0], tex_max[1]], [tex_max[0], tex_min[1]], [tex_min[0], tex_min[1]]);
        } else {
            self.texture_quad(index, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        }
        index
    }

    pub fn create_cuboid(&mut self, pos_min: [f32; 3], pos_max: [f32; 3]) -> u32
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> {

        let index = 
        self.create_box_face(AxisDirection::NegX, [pos_min[1], pos_min[2]], [pos_max[1], pos_max[2]], -pos_min[0]);
        self.create_box_face(AxisDirection::PosX, [pos_min[1], pos_min[2]], [pos_max[1], pos_max[2]], pos_max[0]);
        self.create_box_face(AxisDirection::NegY, [pos_min[0], pos_min[2]], [pos_max[0], pos_max[2]], -pos_min[1]);
        self.create_box_face(AxisDirection::PosY, [pos_min[0], pos_min[2]], [pos_max[0], pos_max[2]], pos_max[1]);
        self.create_box_face(AxisDirection::NegZ, [pos_min[0], pos_min[1]], [pos_max[0], pos_max[1]], -pos_min[2]);
        self.create_box_face(AxisDirection::PosZ, [pos_min[0], pos_min[1]], [pos_max[0], pos_max[1]], pos_max[2]);
        index
    }

    pub fn create_cuboid_textured(&mut self, pos_min: [f32; 3], pos_max: [f32; 3], tex_min: [f32; 2], tex_max: [f32; 2]) -> u32
    where V: VertexHasPosition<f32> + VertexHasNormal<f32> + VertexHasTexture<f32> {
        
        let index = self.create_cuboid(pos_min, pos_max);
        
        self.texture_quad(index, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(index + 4, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(index + 8, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(index + 12, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(index + 16, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        self.texture_quad(index + 20, [tex_min[0], tex_min[1]], [tex_max[0], tex_min[1]], [tex_max[0], tex_max[1]], [tex_min[0], tex_max[1]]);
        
        index
    }
}





pub trait VertexHasPosition<T>: Default {
    fn position(&self) -> &[T; 3];
    fn set_position(&mut self, pos: [T; 3]);
}

pub trait VertexHasNormal<T>: Default {
    fn normal(&self) -> &[T; 3];
    fn set_normal(&mut self, normal: [T; 3]);
}

pub trait VertexHasColour<T>: Default {
    fn colour(&self) -> &[T; 3];
    fn set_colour(&mut self, colour: [T; 3]);
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
    pub fn colour<T>(mut self, colour: [T; 3]) -> Self
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
            vertices: vec![],
            indices: vec![],
        }
    }
}