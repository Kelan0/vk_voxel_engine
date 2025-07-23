use crate::core::{DebugRenderContext, MeshPrimitiveType, Transform};
use anyhow::Result;
use glam::{U8Vec4, Vec3};

pub struct AxisAlignedBoundingBox {
    pub min_coord: Vec3,
    pub max_coord: Vec3
}

impl AxisAlignedBoundingBox {
    pub fn new(min_extent: Vec3, max_extent: Vec3) -> Self {
        AxisAlignedBoundingBox {
            min_coord: min_extent,
            max_coord: max_extent
        }
    }
}

impl BoundingVolume for AxisAlignedBoundingBox {
    fn center(&self) -> Vec3 {
        (self.min_coord + self.max_coord) * 0.5
    }

    fn half_extent(&self) -> Vec3 {
        self.extent() * 0.5
    }

    fn extent(&self) -> Vec3 {
        self.max_coord - self.min_coord
    }

    fn min_coord(&self) -> Vec3 {
        self.min_coord
    }

    fn max_coord(&self) -> Vec3 {
        self.max_coord
    }
}

impl BoundingVolumeDebugDraw for AxisAlignedBoundingBox {
    fn draw_debug(&self, ctx: &mut DebugRenderContext) -> Result<()> {
        ctx.add_mesh(debug_mesh::mesh_box_lines(), *Transform::new()
            .set_translation(self.center())
            .set_scale(self.extent()),
        U8Vec4::new(255, 200, 128, 255));
        Ok(())
    }
}


pub trait BoundingVolume {
    fn center(&self) -> Vec3;

    fn half_extent(&self) -> Vec3;

    fn extent(&self) -> Vec3;

    fn min_coord(&self) -> Vec3;

    fn max_coord(&self) -> Vec3;
}

pub trait BoundingVolumeDebugDraw {
    fn draw_debug(&self, ctx: &mut DebugRenderContext) -> Result<()>;
}

pub mod debug_mesh {
    use crate::core::{BaseVertex, GraphicsManager, Mesh, MeshData, MeshPrimitiveType};
    use anyhow::Result;
    use lazy_static::lazy_static;
    use std::sync::{Arc, Mutex};
    use glam::Vec3;
    use crate::core::world::CHUNK_BOUNDS;

    lazy_static! {
        // dirty mutable global state...
        static ref MESH_BOX_LINES: Mutex<Option<Arc<Mesh<BaseVertex>>>> = Mutex::new(None);
        static ref MESH_GRID_32_LINES: Mutex<Option<Arc<Mesh<BaseVertex>>>> = Mutex::new(None);
    }

    pub fn init(graphics: &GraphicsManager) -> Result<()> {
        let allocator = graphics.memory_allocator();
        let mut cmd_buf = graphics.begin_transfer_commands()?;

        let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::LineList);
        mesh_data.create_cuboid_textured([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0], [1.0, 1.0]);
        let staging_buffer = mesh_data.create_staging_buffer(allocator.clone())?;
        let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;
        MESH_BOX_LINES.lock().expect("Failed to lock MESH_BOX_LINES").replace(Arc::new(mesh));


        let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::LineList);
        mesh_data.create_lines_grid([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0], [32, 32]);
        let staging_buffer = mesh_data.create_staging_buffer(allocator.clone())?;
        let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;
        MESH_GRID_32_LINES.lock().expect("Failed to lock MESH_GRID_32_LINES").replace(Arc::new(mesh));

        graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;

        Ok(())
    }

    pub fn mesh_box_lines() -> Arc<Mesh<BaseVertex>> {
        MESH_BOX_LINES.lock().expect("Failed to lock MESH_BOX_LINES").as_ref().unwrap().clone()
    }

    pub fn mesh_grid_32_lines() -> Arc<Mesh<BaseVertex>> {
        MESH_GRID_32_LINES.lock().expect("Failed to lock MESH_BOX_LINES").as_ref().unwrap().clone()
    }
}