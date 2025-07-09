use glam::{EulerRot, Mat3, Mat4, Quat, Vec3, Vec4};
use vulkano::buffer::{BufferContents, BufferWriteGuard};

#[derive(Clone, PartialEq)]
pub struct Camera {
    position: Vec3,
    orientation: Quat,
    fov: f32,
    aspect_ratio: f32,
    near_plane: f32,
    far_plane: f32,
    projection_changed: bool,

    view_matrix: Mat4,
    projection_matrix: Mat4,
    view_projection_matrix: Mat4,
}

impl Camera {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_perspective(fov: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) -> Self{
        Camera{
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
                self.projection_matrix = Mat4::perspective_infinite_lh(self.fov.to_radians(), self.aspect_ratio, self.near_plane);
            } else {
                self.projection_matrix = Mat4::perspective_lh(self.fov.to_radians(), self.aspect_ratio, self.near_plane, self.far_plane);
            }
        }

        self.view_matrix = Mat4::from_rotation_translation(self.orientation, self.position);
        self.view_matrix = self.view_matrix.inverse();
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn move_position(&mut self, delta_position: Vec3) {
        self.position += delta_position;
    }

    pub fn set_orientation(&mut self, orientation: Quat) {
        self.orientation = orientation;
    }

    pub fn add_rotation(&mut self, rotation: Quat) {
        self.orientation = rotation * self.orientation;
    }

    pub fn set_rotation_euler(&mut self, pitch: f32, yaw: f32, roll: f32) {
        self.set_orientation(Quat::from_euler(EulerRot::XYZEx, pitch.to_radians(), yaw.to_radians(), roll.to_radians()));
    }

    pub fn add_rotation_euler(&mut self, pitch: f32, yaw: f32, roll: f32) {
        self.add_rotation(Quat::from_euler(EulerRot::XYZEx, pitch.to_radians(), yaw.to_radians(), roll.to_radians()));
    }

    pub fn look_at(&mut self, center: Vec3, up: Vec3) {
        self.set_orientation(Quat::look_at_lh(self.position, center, up))
    }

    pub fn look_to(&mut self, dir: Vec3, up: Vec3) {
        self.set_orientation(Quat::look_to_lh(dir, up))
    }

    pub fn set_pitch(&mut self, pitch: f32) {
        let (_, yaw, roll) = self.euler_angles();
        self.set_orientation(Quat::from_euler(EulerRot::XYZEx, pitch.to_radians(), yaw.to_radians(), roll.to_radians()));
    }

    pub fn set_yaw(&mut self, yaw: f32) {
        let (pitch, _, roll) = self.euler_angles();
        self.set_orientation(Quat::from_euler(EulerRot::XYZEx, pitch.to_radians(), yaw.to_radians(), roll.to_radians()));
    }

    pub fn set_roll(&mut self, roll: f32) {
        let (pitch, yaw, _) = self.euler_angles();
        self.set_orientation(Quat::from_euler(EulerRot::XYZEx, pitch.to_radians(), yaw.to_radians(), roll.to_radians()));
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

    pub fn set_perspective(&mut self, fov: f32, aspect_ratio: f32, near_plane: f32, far_plane: f32) {
        self.set_fov(fov);
        self.set_aspect_ratio(aspect_ratio);
        self.set_near_far_plane(near_plane, far_plane);
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn orientation(&self) -> Quat {
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

    pub fn rotation_mat(&self) -> Mat3 {
        Mat3::from_quat(self.orientation)
    }

    pub fn x_axis(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    pub fn y_axis(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    pub fn z_axis(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    pub fn euler_angles(&self) -> (f32, f32, f32) {
        let (pitch, yaw, roll) = self.orientation.to_euler(EulerRot::XYZEx);
        (pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees())
    }

    pub fn pitch(&self) -> f32 {
        let (pitch, _, _) = self.orientation.to_euler(EulerRot::XYZEx);
        pitch.to_degrees()

        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let roll = f32::atan2(2.0 * (q_w * q_x + q_y * q_z), 1.0 - 2.0 * (q_x * q_x + q_y * q_y));
        // roll.to_degrees()
    }

    pub fn yaw(&self) -> f32 {
        let (_, yaw, _) = self.orientation.to_euler(EulerRot::XYZEx);
        yaw.to_degrees()

        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let pitch = f32::asin(2.0 * (q_w * q_y - q_z * q_x));
        // pitch.to_degrees()
    }

    pub fn roll(&self) -> f32 {
        let (_, _, roll) = self.orientation.to_euler(EulerRot::XYZEx);
        roll.to_degrees()
        
        // let ( q_w, q_x, q_y, q_z ) = (self.orientation.w, self.orientation.x, self.orientation.y, self.orientation.z);
        // let yaw = f32::atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z));
        // yaw.to_degrees()
    }

    pub fn update_camera_buffer(&self, camera_buffer: &mut CameraBufferUBO) {
        camera_buffer.set_view_matrix(&self.view_matrix);
        camera_buffer.set_projection_matrix(&self.projection_matrix);
        camera_buffer.set_view_projection_matrix(&self.view_projection_matrix);
    }

    pub fn get_camera_buffer(&self) -> CameraBufferUBO {
        let mut camera_buffer = CameraBufferUBO::default();
        self.update_camera_buffer(&mut camera_buffer);
        camera_buffer
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::new(0.0, 0.0, 0.0),
            orientation: Quat::IDENTITY,
            fov: 60.0,
            aspect_ratio: 4.0 / 3.0,
            near_plane: 0.01,
            far_plane: 100.0,
            projection_changed: true,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection_matrix: Mat4::IDENTITY,
        }
    }
}

#[derive(BufferContents)]
#[repr(C)]
pub struct CameraBufferUBO {
    pub view_matrix: [f32; 16],
    pub projection_matrix: [f32; 16],
    pub view_projection_matrix: [f32; 16],
}

impl CameraBufferUBO {
    pub fn new() -> Self {
        let mut val = CameraBufferUBO::default();
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

impl Default for CameraBufferUBO {
    fn default() -> Self {
        CameraBufferUBO {
            view_matrix: [0.0; 16],
            projection_matrix: [0.0; 16],
            view_projection_matrix: [0.0; 16],
        }
    }
}
