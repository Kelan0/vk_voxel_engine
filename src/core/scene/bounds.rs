use crate::core::{DebugRenderContext, MeshPrimitiveType, Transform};
use anyhow::Result;
use glam::{DVec3, U8Vec4, Vec3};

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct AxisAlignedBoundingBox {
    pub min_coord: DVec3,
    pub max_coord: DVec3
}

impl AxisAlignedBoundingBox {
    pub fn new(min_extent: DVec3, max_extent: DVec3) -> Self {
        AxisAlignedBoundingBox {
            min_coord: min_extent,
            max_coord: max_extent
        }
    }
}

impl BoundingVolume for AxisAlignedBoundingBox {
    fn center(&self) -> DVec3 {
        (self.min_coord + self.max_coord) * 0.5
    }

    fn half_extent(&self) -> DVec3 {
        self.extent() * 0.5
    }

    fn extent(&self) -> DVec3 {
        self.max_coord - self.min_coord
    }

    fn min_coord(&self) -> DVec3 {
        self.min_coord
    }

    fn max_coord(&self) -> DVec3 {
        self.max_coord
    }

    fn closest_point(&self, pos: DVec3) -> DVec3 {
        DVec3::new(
            pos.x.clamp(self.min_coord.x, self.max_coord.x),
            pos.y.clamp(self.min_coord.y, self.max_coord.y),
            pos.z.clamp(self.min_coord.z, self.max_coord.z)
        )
    }
    
    #[allow(refining_impl_trait)]
    fn translate(&self, translation: DVec3) -> AxisAlignedBoundingBox {
        AxisAlignedBoundingBox {
            min_coord: self.min_coord + translation,
            max_coord: self.max_coord + translation,
        }
    }
}

impl BoundingVolumeDebugDraw for AxisAlignedBoundingBox {
    fn draw_debug(&self, ctx: &mut DebugRenderContext) -> Result<()> {
        ctx.add_mesh(debug_mesh::mesh_box_lines(), *Transform::new()
            .set_translation(self.center().as_vec3())
            .set_scale(self.extent().as_vec3()),
        U8Vec4::new(255, 200, 128, 255));
        Ok(())
    }
}


pub trait BoundingVolume {
    fn center(&self) -> DVec3;

    fn half_extent(&self) -> DVec3;

    fn extent(&self) -> DVec3;

    fn min_coord(&self) -> DVec3;

    fn max_coord(&self) -> DVec3;

    fn closest_point(&self, pos: DVec3) -> DVec3;

    fn translate(&self, translation: DVec3) -> impl BoundingVolume;
}

pub trait BoundingVolumeDebugDraw {
    fn draw_debug(&self, ctx: &mut DebugRenderContext) -> Result<()>;
}

pub mod debug_mesh {
    use std::any::Any;
    use std::collections::HashMap;
    use crate::core::{util, BaseVertex, GraphicsManager, Mesh, MeshData, MeshDataConfig, MeshPrimitiveType};
    use anyhow::Result;
    use lazy_static::lazy_static;
    use std::sync::{Arc, Mutex};

    lazy_static! {
        // dirty mutable global state...
        static ref MESH_BOX_LINES: Mutex<Option<Arc<Mesh<BaseVertex>>>> = Mutex::new(None);
        static ref MESH_GRID_LINES: Mutex<Option<HashMap<u32, Arc<Mesh<BaseVertex>>>>> = Mutex::new(None);
    }

    pub fn init(graphics: &GraphicsManager) -> Result<()> {
        let allocator = graphics.memory_allocator();
        let mut cmd_buf = graphics.begin_transfer_commands()?;

        let mut mesh_data = MeshData::<BaseVertex>::new(MeshDataConfig::new(MeshPrimitiveType::LineList));
        mesh_data.create_cuboid_textured([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], util::f32_to_u16_norm([0.0, 0.0]), util::f32_to_u16_norm([1.0, 1.0]));
        mesh_data.colour_vertices(.., [255, 255, 255, 255]);
        let staging_buffer = mesh_data.create_staging_buffer(allocator.clone())?;
        let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;
        MESH_BOX_LINES.lock().expect("Failed to lock MESH_BOX_LINES").replace(Arc::new(mesh));


        // keep allocated staging buffers until after the transfer commands have completed.
        let mut temp_resources: Vec<Arc<dyn Any>> = vec![];

        let mut mesh_grid_lines = HashMap::new();

        for i in 2..=5 {
            let dim = 1 << i;

            let mut mesh_data = MeshData::<BaseVertex>::new(MeshDataConfig::new(MeshPrimitiveType::LineList));
            mesh_data.create_lines_grid([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0], [dim, dim]);
            mesh_data.colour_vertices(.., [255, 255, 255, 255]);
            let staging_buffer = mesh_data.create_staging_buffer(allocator.clone())?;
            let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;

            temp_resources.push(staging_buffer.buffer().clone());

            mesh_grid_lines.insert(dim, Arc::new(mesh));
        }
        MESH_GRID_LINES.lock().expect("Failed to lock MESH_GRID_LINES").replace(mesh_grid_lines);


        graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;

        temp_resources.clear();
        Ok(())
    }

    pub fn mesh_box_lines() -> Arc<Mesh<BaseVertex>> {
        MESH_BOX_LINES.lock()
            .expect("Failed to lock MESH_BOX_LINES")
            .as_ref()
            .expect("MESH_BOX_LINES is not initialized")
            .clone()
    }

    pub fn mesh_grid_lines(dim: u32) -> Arc<Mesh<BaseVertex>> {
        MESH_GRID_LINES.lock()
            .expect("Failed to lock MESH_GRID_LINES")
            .as_ref()
            .expect("MESH_GRID_LINES is not initialized")
            .get(&dim)
            .expect(format!("MESH_GRID_LINES has no mesh for dimension {dim}").as_str())
            .clone()
    }
}