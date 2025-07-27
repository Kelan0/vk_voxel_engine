use glam::{DMat3, DMat4, DQuat, DVec3, EulerRot, Mat3, Mat4, Quat, Vec3};
use vulkano::buffer::BufferContents;

#[derive(Clone, PartialEq)]
pub struct Camera {
    position: DVec3,
    orientation: DQuat,
    fov: f32,
    aspect_ratio: f32,
    near_plane: f32,
    far_plane: f32,
    projection_changed: bool,

    view_matrix: DMat4,
    projection_matrix: Mat4,
    view_projection_matrix: DMat4,
}

impl Camera {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_perspective(fov: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self {
        Camera {
            fov,
            aspect_ratio,
            near_plane,
            far_plane,
            ..Default::default()
        }
    }

    pub fn update(&mut self) {
        if self.projection_changed {
            if self.is_infinite_far_plane() {
                self.projection_matrix = Mat4::perspective_infinite_lh(
                    self.fov.to_radians(),
                    self.aspect_ratio,
                    self.near_plane,
                );
            } else {
                self.projection_matrix = Mat4::perspective_lh(
                    self.fov.to_radians(),
                    self.aspect_ratio,
                    self.near_plane,
                    self.far_plane,
                );
            }
        }

        self.view_matrix = DMat4::from_rotation_translation(self.orientation, self.position / 32.0);
        self.view_matrix = self.view_matrix.inverse();
        self.view_projection_matrix = self.projection_matrix.as_dmat4() * self.view_matrix;
    }

    pub fn set_position(&mut self, position: DVec3) {
        self.position = position;
    }

    pub fn move_position(&mut self, delta_position: DVec3) {
        self.position += delta_position;
    }

    pub fn set_orientation(&mut self, orientation: DQuat) {
        self.orientation = orientation;
    }

    pub fn add_rotation(&mut self, rotation: DQuat) {
        self.orientation = rotation * self.orientation;
    }

    pub fn set_rotation_euler(&mut self, pitch: f64, yaw: f64, roll: f64) {
        self.set_orientation(DQuat::from_euler(
            EulerRot::XYZEx,
            pitch.to_radians(),
            yaw.to_radians(),
            roll.to_radians(),
        ));
    }

    pub fn add_rotation_euler(&mut self, pitch: f64, yaw: f64, roll: f64) {
        self.add_rotation(DQuat::from_euler(
            EulerRot::XYZEx,
            pitch.to_radians(),
            yaw.to_radians(),
            roll.to_radians(),
        ));
    }

    pub fn look_at(&mut self, center: DVec3, up: DVec3) {
        self.set_orientation(DQuat::look_at_lh(self.position, center, up))
    }

    pub fn look_to(&mut self, dir: DVec3, up: DVec3) {
        self.set_orientation(DQuat::look_to_lh(dir, up))
    }

    pub fn set_pitch(&mut self, pitch: f64) {
        let (_, yaw, roll) = self.euler_angles();
        self.set_orientation(DQuat::from_euler(
            EulerRot::XYZEx,
            pitch.to_radians(),
            yaw.to_radians(),
            roll.to_radians(),
        ));
    }

    pub fn set_yaw(&mut self, yaw: f64) {
        let (pitch, _, roll) = self.euler_angles();
        self.set_orientation(DQuat::from_euler(
            EulerRot::XYZEx,
            pitch.to_radians(),
            yaw.to_radians(),
            roll.to_radians(),
        ));
    }

    pub fn set_roll(&mut self, roll: f64) {
        let (pitch, yaw, _) = self.euler_angles();
        self.set_orientation(DQuat::from_euler(
            EulerRot::XYZEx,
            pitch.to_radians(),
            yaw.to_radians(),
            roll.to_radians(),
        ));
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
        self.projection_changed = true;
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.projection_changed = true;
    }

    pub fn set_near_far_plane(&mut self, near_plane: f32, far_plane: f32) {
        self.near_plane = near_plane;
        self.far_plane = far_plane;
        self.projection_changed = true;
    }

    pub fn set_perspective(
        &mut self,
        fov: f32,
        aspect_ratio: f32,
        near_plane: f32,
        far_plane: f32,
    ) {
        self.set_fov(fov);
        self.set_aspect_ratio(aspect_ratio);
        self.set_near_far_plane(near_plane, far_plane);
    }

    pub fn position(&self) -> DVec3 {
        self.position
    }

    pub fn orientation(&self) -> DQuat {
        self.orientation
    }

    pub fn fov(&self) -> f32 {
        self.fov
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.aspect_ratio
    }

    pub fn near_plane(&self) -> f32 {
        self.near_plane
    }

    pub fn far_plane(&self) -> f32 {
        self.far_plane
    }

    pub fn is_infinite_far_plane(&self) -> bool {
        self.far_plane.is_infinite()
    }

    pub fn rotation_mat(&self) -> DMat3 {
        DMat3::from_quat(self.orientation)
    }

    pub fn x_axis(&self) -> DVec3 {
        self.orientation * DVec3::X
    }

    pub fn y_axis(&self) -> DVec3 {
        self.orientation * DVec3::Y
    }

    pub fn z_axis(&self) -> DVec3 {
        self.orientation * DVec3::Z
    }

    pub fn euler_angles(&self) -> (f64, f64, f64) {
        let (pitch, yaw, roll) = self.orientation.to_euler(EulerRot::XYZEx);
        (pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees())
    }

    pub fn pitch(&self) -> f64 {
        let (pitch, _, _) = self.orientation.to_euler(EulerRot::XYZEx);
        pitch.to_degrees()

        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let roll = f32::atan2(2.0 * (q_w * q_x + q_y * q_z), 1.0 - 2.0 * (q_x * q_x + q_y * q_y));
        // roll.to_degrees()
    }

    pub fn yaw(&self) -> f64 {
        let (_, yaw, _) = self.orientation.to_euler(EulerRot::XYZEx);
        yaw.to_degrees()

        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let pitch = f32::asin(2.0 * (q_w * q_y - q_z * q_x));
        // pitch.to_degrees()
    }

    pub fn roll(&self) -> f64 {
        let (_, _, roll) = self.orientation.to_euler(EulerRot::XYZEx);
        roll.to_degrees()

        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let yaw = f32::atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z));
        // yaw.to_degrees()
    }

    pub fn update_camera_buffer(&self, camera_buffer: &mut CameraDataUBO) {
        camera_buffer.set_view_matrix(&self.view_matrix.as_mat4());
        camera_buffer.set_projection_matrix(&self.projection_matrix);
        camera_buffer.set_view_projection_matrix(&self.view_projection_matrix.as_mat4());
    }

    pub fn get_camera_buffer(&self) -> CameraDataUBO {
        let mut camera_buffer = CameraDataUBO::default();
        self.update_camera_buffer(&mut camera_buffer);
        camera_buffer
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            fov: 60.0,
            aspect_ratio: 4.0 / 3.0,
            near_plane: 0.01,
            far_plane: 100.0,
            projection_changed: true,
            view_matrix: DMat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection_matrix: DMat4::IDENTITY,
        }
    }
}

#[derive(BufferContents)]
#[repr(C)]
pub struct CameraDataUBO {
    pub view_matrix: [f32; 16],
    pub projection_matrix: [f32; 16],
    pub view_projection_matrix: [f32; 16],
}

impl CameraDataUBO {
    pub fn new() -> Self {
        let mut val = CameraDataUBO::default();
        val.set_view_matrix(&Mat4::IDENTITY);
        val.set_projection_matrix(&Mat4::IDENTITY);
        val.set_view_projection_matrix(&Mat4::IDENTITY);
        val
    }

    pub fn set_view_matrix(&mut self, mat: &Mat4) {
        mat.write_cols_to_slice(&mut self.view_matrix);
    }

    pub fn set_projection_matrix(&mut self, mat: &Mat4) {
        mat.write_cols_to_slice(&mut self.projection_matrix);
    }

    pub fn set_view_projection_matrix(&mut self, mat: &Mat4) {
        mat.write_cols_to_slice(&mut self.view_projection_matrix);
    }
}

impl Default for CameraDataUBO {
    fn default() -> Self {
        CameraDataUBO {
            view_matrix: [0.0; 16],
            projection_matrix: [0.0; 16],
            view_projection_matrix: [0.0; 16],
        }
    }
}
