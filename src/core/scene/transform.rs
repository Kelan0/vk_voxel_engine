use bevy_ecs::component::Component;
use glam::{Affine3A, Mat3, Mat3A, Mat4, Quat, Vec3, Vec3A};

#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub affine: Affine3A,
}

#[allow(clippy::derivable_impls)]
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

    const SCALE: f32 = 1.0 / 32.0;

    pub fn new() -> Self {
        // Transform { affine }
        Default::default()
    }

    pub fn transform_global(&mut self, affine: Affine3A) -> &mut Transform {
        self.affine = affine * self.affine;
        self
    }

    pub fn transform(&mut self, affine: Affine3A) -> &mut Transform {
        self.affine *= affine;
        self
    }

    pub fn set_translation_a(&mut self, translation: Vec3A) -> &mut Self {
        self.affine.translation = translation;
        self
    }

    pub fn set_translation(&mut self, translation: Vec3) -> &mut Self {
        self.set_translation_a(translation.into())
    }

    pub fn translate_global_a(&mut self, translation: Vec3A) -> &mut Self {
        self.transform_global(Affine3A { matrix3: Mat3A::IDENTITY, translation })
    }

    pub fn translate_a(&mut self, translation: Vec3A) -> &mut Self {
        self.transform(Affine3A { matrix3: Mat3A::IDENTITY, translation })
    }

    pub fn translate_global(&mut self, translation: Vec3) -> &mut Self {
        self.transform_global(Affine3A::from_translation(translation))
    }

    pub fn translate(&mut self, translation: Vec3) -> &mut Self {
        self.transform(Affine3A::from_translation(translation))
    }

    pub fn set_rotation_mat_a(&mut self, rotation: Mat3A) -> &mut Self {
        self.affine.matrix3 = rotation;
        self
    }

    pub fn set_rotation_mat(&mut self, rotation: Mat3) -> &mut Self {
        self.set_rotation_mat_a(rotation.into())
    }

    pub fn rotate_mat_a(&mut self, rotation: Mat3A) -> &mut Self {
        self.transform_global(Affine3A { matrix3: rotation, translation: Vec3A::ZERO })
    }

    pub fn rotate_mat(&mut self, rotation: Mat3) -> &mut Self {
        self.transform_global(Affine3A::from_mat3(rotation))
    }

    pub fn set_rotation_quat(&mut self, rotation: Quat) -> &mut Self {
        self.affine.matrix3 = Mat3A::from_quat(rotation);
        self
    }

    pub fn rotate_quat(&mut self, rotation: Quat) -> &mut Self {
        self.transform_global(Affine3A::from_quat(rotation))
    }

    pub fn rotate_axis_angle(&mut self, axis: Vec3, angle: f32) -> &mut Self {
        self.transform_global(Affine3A::from_axis_angle(axis, angle))
    }

    pub fn rotate_global_x(&mut self, angle: f32) -> &mut Self {
        self.transform_global(Affine3A::from_rotation_x(angle))
    }

    pub fn rotate_global_y(&mut self, angle: f32) -> &mut Self {
        self.transform_global(Affine3A::from_rotation_y(angle))
    }

    pub fn rotate_global_z(&mut self, angle: f32) -> &mut Self {
        self.transform_global(Affine3A::from_rotation_z(angle))
    }

    pub fn rotate_x(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A::from_rotation_x(angle))
    }

    pub fn rotate_y(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A::from_rotation_y(angle))
    }

    pub fn rotate_z(&mut self, angle: f32) -> &mut Self {
        self.transform(Affine3A::from_rotation_z(angle))
    }

    pub fn scale(&mut self, scale: Vec3) -> &mut Transform {
        self.transform(Affine3A::from_scale(scale))
    }

    pub fn set_scale(&mut self, scale: Vec3) -> &mut Transform {
        let mut rotation = self.get_rotation_mat3(true);
        rotation.x_axis *= scale.x;
        rotation.y_axis *= scale.x;
        rotation.z_axis *= scale.x;
        self.set_rotation_mat(rotation)
    }

    pub fn get_translation(&self) -> Vec3 {
        self.affine.translation.into()
    }

    pub fn get_rotation_mat3(&self, safe: bool) -> Mat3 {
        self.get_rotation_mat3_scale(safe).0
    }

    pub fn get_rotation_mat3_scale(&self, safe: bool) -> (Mat3, Vec3) {
        let scale = self.get_scale();
        let rot = if safe && f32::abs(scale.x * scale.y * scale.z) < 1e-15 {
            // Scale is close to zero on one or more axis, so we cannot determine the rotation here.
            Mat3::IDENTITY
        } else {
            let inv_scale = scale.recip();
            Mat3::from_cols(
                Vec3::from(self.affine.matrix3.x_axis) * inv_scale.x,
                Vec3::from(self.affine.matrix3.y_axis) * inv_scale.y,
                Vec3::from(self.affine.matrix3.z_axis) * inv_scale.z
            )
        };
        (rot, scale)
    }

    pub fn get_rotation_quat(&self, safe: bool) -> Quat {
        let rot = self.get_rotation_mat3(safe);
        Quat::from_mat3(&rot)
    }

    pub fn get_rotation_quat_scale(&self, safe: bool) -> (Quat, Vec3) {
        let (rot, scale) = self.get_rotation_mat3_scale(safe);
        (Quat::from_mat3(&rot), scale)
    }

    pub fn get_scale(&self) -> Vec3 {
        let det = self.affine.matrix3.determinant();
        Vec3::new(
            self.affine.matrix3.x_axis.length() * f32::signum(det),
            self.affine.matrix3.y_axis.length(),
            self.affine.matrix3.z_axis.length(),
        )
    }

    pub fn get_translation_rotation_quat_scale(&self, safe: bool) -> (Vec3, Quat, Vec3) {
        let translation = self.get_translation();
        let (rotation, scale) = self.get_rotation_quat_scale(safe);
        (translation, rotation, scale)
    }

    pub fn get_translation_rotation_mat3_scale(&self, safe: bool) -> (Vec3, Mat3, Vec3) {
        let translation = self.get_translation();
        let (rotation, scale) = self.get_rotation_mat3_scale(safe);
        (translation, rotation, scale)
    }

    pub fn get_mat4(&self) -> Mat4 {
        Mat4::from_cols(
            self.affine.matrix3.x_axis.extend(0.0),
            self.affine.matrix3.y_axis.extend(0.0),
            self.affine.matrix3.z_axis.extend(0.0),
            (self.affine.translation * Self::SCALE).extend(1.0),
        )
    }

    pub fn write_model_matrix(&self, matrix_data: &mut [f32]) {
        debug_assert!(matrix_data.len() >= 16);

        self.get_mat4().write_cols_to_slice(matrix_data);

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
