use crate::core::{DebugRenderContext, Transform};
use anyhow::Result;
use glam::Vec3;

pub struct AxisAlignedBoundingBox {
    pub min_extent: Vec3,
    pub max_extent: Vec3
}

impl AxisAlignedBoundingBox {
    pub fn new(min_extent: Vec3, max_extent: Vec3) -> Self {
        AxisAlignedBoundingBox {
            min_extent, max_extent
        }
    }
}

impl BoundingVolume for AxisAlignedBoundingBox {
    fn center(&self) -> Vec3 {
        (self.min_extent + self.max_extent) * 0.5
    }

    fn half_extent(&self) -> Vec3 {
        (self.max_extent - self.min_extent) * 0.5
    }

    fn min_extent(&self) -> Vec3 {
        self.min_extent
    }

    fn max_extent(&self) -> Vec3 {
        self.max_extent
    }
}

impl BoundingVolumeDebugDraw for AxisAlignedBoundingBox {
    fn draw(&self, ctx: &mut DebugRenderContext) -> Result<()> {
        ctx.add_mesh(debug_mesh::mesh_box_lines(), *Transform::new()
            .set_translation(self.center())
            .set_scale(self.half_extent()));
        Ok(())
    }
}


pub trait BoundingVolume {
    fn center(&self) -> Vec3;

    fn half_extent(&self) -> Vec3;

    fn min_extent(&self) -> Vec3;

    fn max_extent(&self) -> Vec3;
}

pub trait BoundingVolumeDebugDraw {
    fn draw(&self, ctx: &mut DebugRenderContext) -> Result<()>;
}

pub mod debug_mesh {
    use crate::core::{BaseVertex, GraphicsManager, Mesh, MeshData, MeshPrimitiveType};
    use anyhow::Result;
    use lazy_static::lazy_static;
    use std::sync::Mutex;

    lazy_static! {
        // dirty mutable global state...
        static ref MESH_BOX_LINES: Mutex<Option<Mesh<BaseVertex>>> = Mutex::new(None);
    }

    pub fn init(graphics: &GraphicsManager) -> Result<()> {
        let allocator = graphics.memory_allocator();
        let mut cmd_buf = graphics.begin_transfer_commands()?;

        let mut mesh_data = MeshData::<BaseVertex>::new(MeshPrimitiveType::LineList);
        mesh_data.create_cuboid_textured([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.0, 0.0], [1.0, 1.0]);

        let staging_buffer = mesh_data.create_staging_buffer(allocator.clone())?;

        let mesh = mesh_data.build_mesh_staged(allocator.clone(), &mut cmd_buf, &staging_buffer)?;
        MESH_BOX_LINES.lock().expect("Failed to lock MESH_BOX_LINES").replace(mesh);

        graphics.submit_transfer_commands(cmd_buf)?
            .wait(None)?;

        Ok(())
    }

    pub fn mesh_box_lines() -> Mesh<BaseVertex> {
        MESH_BOX_LINES.lock().expect("Failed to lock MESH_BOX_LINES").as_ref().unwrap().clone()
    }
}