use std::ops::Index;
use glam::Vec3;
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(Clone, PartialEq)]
pub struct MeshData<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
    default_colour: [f32; 3],
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
    
    pub fn create_quad(&mut self, v00: V, v01: V, v11: V, v10: V) -> u32 {
        let i00 = self.add_vertex(v00);
        let i01 = self.add_vertex(v01);
        let i11 = self.add_vertex(v11);
        let i10 = self.add_vertex(v10);
        self.add_quad(i00, i01, i11, i10)
    }

    fn create_triangle_primitive(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        let index = self.indices.len() as u32;
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
        index
    }
}

impl <V: Vertex + Default> MeshData<V> {
    pub fn vertex(&mut self) -> VertexBuilder<V> {
        VertexBuilder::new(self)
    }
}

impl <V: Vertex + Default> MeshData<V> {

    pub fn create_cuboid(&mut self, pos_min: [f32; 3], pos_max: [f32; 3])
        where V: VertexHasPosition<f32> + VertexHasNormal<f32> {

        // -X face
        let v00 = self.vertex().pos([pos_min[0], pos_min[1], pos_min[2]]).normal([-1.0, 0.0, 0.0]).get();
        let v01 = self.vertex().pos([pos_min[0], pos_max[1], pos_min[2]]).normal([-1.0, 0.0, 0.0]).get();
        let v11 = self.vertex().pos([pos_min[0], pos_max[1], pos_max[2]]).normal([-1.0, 0.0, 0.0]).get();
        let v10 = self.vertex().pos([pos_min[0], pos_min[1], pos_max[2]]).normal([-1.0, 0.0, 0.0]).get();
        self.create_quad(v00, v01, v11, v10);

        // +X face
        let v00 = self.vertex().pos([pos_max[0], pos_min[1], pos_min[2]]).normal([1.0, 0.0, 0.0]).get();
        let v01 = self.vertex().pos([pos_max[0], pos_min[1], pos_max[2]]).normal([1.0, 0.0, 0.0]).get();
        let v11 = self.vertex().pos([pos_max[0], pos_max[1], pos_max[2]]).normal([1.0, 0.0, 0.0]).get();
        let v10 = self.vertex().pos([pos_max[0], pos_max[1], pos_min[2]]).normal([1.0, 0.0, 0.0]).get();
        self.create_quad(v00, v01, v11, v10);

        // -Y face
        let v00 = self.vertex().pos([pos_min[0], pos_min[1], pos_min[2]]).normal([0.0, -1.0, 0.0]).get();
        let v01 = self.vertex().pos([pos_min[0], pos_min[1], pos_max[2]]).normal([0.0, -1.0, 0.0]).get();
        let v11 = self.vertex().pos([pos_max[0], pos_min[1], pos_max[2]]).normal([0.0, -1.0, 0.0]).get();
        let v10 = self.vertex().pos([pos_max[0], pos_min[1], pos_min[2]]).normal([0.0, -1.0, 0.0]).get();
        self.create_quad(v00, v01, v11, v10);

        // +Y face
        let v00 = self.vertex().pos([pos_min[0], pos_max[1], pos_min[2]]).normal([0.0, 1.0, 0.0]).get();
        let v01 = self.vertex().pos([pos_max[0], pos_max[1], pos_min[2]]).normal([0.0, 1.0, 0.0]).get();
        let v11 = self.vertex().pos([pos_max[0], pos_max[1], pos_max[2]]).normal([0.0, 1.0, 0.0]).get();
        let v10 = self.vertex().pos([pos_min[0], pos_max[1], pos_max[2]]).normal([0.0, 1.0, 0.0]).get();
        self.create_quad(v00, v01, v11, v10);

        // -Z face
        let v00 = self.vertex().pos([pos_min[0], pos_min[1], pos_min[2]]).normal([0.0, 0.0, -1.0]).get();
        let v01 = self.vertex().pos([pos_max[0], pos_min[1], pos_min[2]]).normal([0.0, 0.0, -1.0]).get();
        let v11 = self.vertex().pos([pos_max[0], pos_max[1], pos_min[2]]).normal([0.0, 0.0, -1.0]).get();
        let v10 = self.vertex().pos([pos_min[0], pos_max[1], pos_min[2]]).normal([0.0, 0.0, -1.0]).get();
        self.create_quad(v00, v01, v11, v10);

        // +Z face
        let v00 = self.vertex().pos([pos_min[0], pos_min[1], pos_max[2]]).normal([0.0, 0.0, 1.0]).get();
        let v01 = self.vertex().pos([pos_min[0], pos_max[1], pos_max[2]]).normal([0.0, 0.0, 1.0]).get();
        let v11 = self.vertex().pos([pos_max[0], pos_max[1], pos_max[2]]).normal([0.0, 0.0, 1.0]).get();
        let v10 = self.vertex().pos([pos_max[0], pos_min[1], pos_max[2]]).normal([0.0, 0.0, 1.0]).get();
        self.create_quad(v00, v01, v11, v10);

        let r: Vec<u32>;
        
        // Continue for rest of the faces
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


impl <V: Vertex> Default for MeshData<V> {
    fn default() -> Self {
        Self{
            vertices: vec![],
            indices: vec![],
            default_colour: [1.0, 1.0, 1.0],
        }
    }
}