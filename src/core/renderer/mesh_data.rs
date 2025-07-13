use vulkano::pipeline::graphics::vertex_input::Vertex;

#[derive(Clone, PartialEq)]
pub struct MeshData<V: Vertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>
    
}

impl <V: Vertex> Default for MeshData<V> {
    fn default() -> Self {
        Self{
            vertices: vec![],
            indices: vec![],
        }
    }
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
        _ = self.create_triangle_primitive(i0, i2, i3);
        i
    }
    
    fn create_triangle_primitive(&mut self, i0: u32, i1: u32, i2: u32) -> u32 {
        let index = self.indices.len() as u32;
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
        index
    }
}