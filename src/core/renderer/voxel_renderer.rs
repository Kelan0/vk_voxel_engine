use glam::{U16Vec2, U8Vec4, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use crate::core::{AxisDirection};

#[derive(BufferContents, Vertex, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct VoxelVertex {
    #[format(R32_UINT)]
    pub vs_data: u32, // XXXXXYYYYYZZZZZNNN // X[5] Y[5] Z[5] Norm[3]
}

impl VoxelVertex {
    const MASK_X: u32 = 0b11111_00000_00000_000;
    const SHIFT_X: u32 = 5+5+3;
    const MASK_Y: u32 = 0b00000_11111_00000_000;
    const SHIFT_Y: u32 = 5+3;
    const MASK_Z: u32 = 0b00000_00000_11111_000;
    const SHIFT_Z: u32 = 3;
    const MASK_NORM: u32 = 0b00000_00000_00000_111;
    const SHIFT_NORM: u32 = 0;

    pub fn from_packed(data: u32) -> Self {
        VoxelVertex {
            vs_data: data
        }
    }

    pub fn from_unpacked(x: u32, y: u32, z: u32, dir: AxisDirection) -> Self {
        let data = Self::pack_data(x, y, z, dir);
        Self::from_packed(data)
    }

    pub fn pack_data(x: u32, y: u32, z: u32, dir: AxisDirection) -> u32 {
        (x & (Self::MASK_X >> Self::SHIFT_X) << Self::SHIFT_X) |
            (y & (Self::MASK_Y >> Self::SHIFT_Y) << Self::SHIFT_Y) |
            (z & (Self::MASK_Z >> Self::SHIFT_Z) << Self::SHIFT_Z) |
            (dir.index() * (Self::MASK_NORM >> Self::SHIFT_NORM) << Self::SHIFT_NORM)
    }

    pub fn unpack_x(data: u32) -> u32 {
        (data & Self::MASK_X) >> Self::SHIFT_X
    }

    pub fn unpack_y(data: u32) -> u32 {
        (data & Self::MASK_Y) >> Self::SHIFT_Y
    }

    pub fn unpack_z(data: u32) -> u32 {
        (data & Self::MASK_Z) >> Self::SHIFT_Z
    }

    pub fn unpack_dir(data: u32) -> u32 {
        (data & Self::MASK_NORM) >> Self::SHIFT_NORM
    }

    pub fn packed_data(&self) -> u32 {
        self.vs_data
    }
}

struct VoxelRenderer {

}

impl VoxelRenderer {

}