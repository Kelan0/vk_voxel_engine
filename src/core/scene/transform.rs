use bevy_ecs::component::Component;
use glam::{Affine3A, Mat3, Mat3A, Mat4, Quat, Vec3, Vec3A};

#[derive(Component, Clone, Copy, PartialEq)]
pub struct Transform {
    pub affine: Affine3A,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            affine: Affine3A::default(),
            // translation: DVec3::ZERO,
            // rotation: Mat3::IDENTITY,
            // scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn new() -> Self {
        // Transform { affine }
        Default::default()
    }

    pub fn transform(&mut self, affine: Affine3A) -> &mut Transform {
        self.affine = affine * self.affine;
        self
    }

    pub fn set_translation_a(&mut self, translation: Vec3A) -> &mut Self {
        self.affine.translation = translation;
        self
    }

    pub fn set_translation(&mut self, translation: Vec3) -> &mut Self {
        self.set_translation_a(translation.into())
    }

    pub fn translate_a(&mut self, translation: Vec3A) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::IDENTITY,
            translation,
        })
    }

    pub fn translate(&mut self, translation: Vec3) -> &mut Self {
        self.translate_a(translation.into())
    }

    pub fn set_rotation_mat_a(&mut self, rotation: Mat3A) -> &mut Self {
        self.affine.matrix3 = rotation;
        self
    }

    pub fn set_rotation_mat(&mut self, rotation: Mat3) -> &mut Self {
        self.set_rotation_mat_a(rotation.into())
    }

    pub fn rotate_mat_a(&mut self, rotation: Mat3A) -> &mut Self {
        self.transform(Affine3A {
            matrix3: rotation,
            translation: Vec3A::ZERO,
        })
    }

    pub fn rotate_mat(&mut self, rotation: Mat3) -> &mut Self {
        self.rotate_mat_a(rotation.into())
    }

    pub fn set_rotation_quat(&mut self, rotation: Quat) -> &mut Self {
        self.affine.matrix3 = Mat3A::from_quat(rotation);
        self
    }

    pub fn rotate_quat(&mut self, rotation: Quat) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::from_quat(rotation),
            translation: Vec3A::ZERO,
        })
    }

    pub fn rotate_axis_angle(&mut self, axis: Vec3, angle: f32) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::from_axis_angle(axis, angle),
            translation: Vec3A::ZERO,
        })
    }

    pub fn rotate_x(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::from_rotation_x(angle),
            translation: Vec3A::ZERO,
        })
    }

    pub fn rotate_y(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::from_rotation_y(angle),
            translation: Vec3A::ZERO,
        })
    }

    pub fn rotate_z(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A {
            matrix3: Mat3A::from_rotation_y(angle),
            translation: Vec3A::ZERO,
        })
    }

    pub fn to_mat4(&self) -> Mat4 {
        self.affine.into()
    }

    pub fn write_model_matrix(&self, matrix_data: &mut [f32]) {
        debug_assert!(matrix_data.len() >= 16);

        self.to_mat4().write_cols_to_slice(matrix_data);

        // // 0 | 4 |  8 | 12
        // // 1 | 5 |  9 | 13
        // // 2 | 6 | 10 | 14
        // // 3 | 7 | 11 | 15
        //
        // self.affine.matrix3.x_axis.write_to_slice(&mut matrix_data[0..3]);
        // self.affine.matrix3.y_axis.write_to_slice(&mut matrix_data[4..7]);
        // self.affine.matrix3.z_axis.write_to_slice(&mut matrix_data[8..11]);
        // self.affine.translation.write_to_slice(&mut matrix_data[12..15]);
        // matrix_data[3] = 0.0;
        // matrix_data[7] = 0.0;
        // matrix_data[11] = 0.0;
        // matrix_data[15] = 1.0;
    }
}
