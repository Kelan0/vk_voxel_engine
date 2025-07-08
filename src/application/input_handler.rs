use glam::Vec2;
use log::warn;
use sdl3::event::Event;
use sdl3::keyboard::Scancode;
use sdl3::mouse::MouseWheelDirection;

const USE_SCANCODES: bool = true;
const KEYBOARD_SIZE: usize = Scancode::Count as usize;
const MOUSE_SIZE: usize = 16;

pub use sdl3::keyboard::Scancode as Key;
pub use sdl3::mouse::MouseButton as MouseBtn;

pub struct InputHandler {
    keys_down: [bool; KEYBOARD_SIZE],
    keys_pressed: [bool; KEYBOARD_SIZE],
    keys_released: [bool; KEYBOARD_SIZE],
    mouse_down: [bool; MOUSE_SIZE],
    mouse_pressed: [bool; MOUSE_SIZE],
    mouse_released: [bool; MOUSE_SIZE],
    mouse_dragged: [bool; MOUSE_SIZE],
    mouse_press_pixel_coord: [Vec2; MOUSE_SIZE],
    mouse_drag_pixel_origin: [Vec2; MOUSE_SIZE],
    curr_mouse_pixel_coord: Vec2,
    prev_mouse_pixel_coord: Vec2,
    curr_mouse_pixel_motion: Vec2,
    prev_mouse_pixel_motion: Vec2,
    scroll_amount: Vec2,
    window_size: Vec2,
}

impl InputHandler {
    pub fn new() -> Self {
        InputHandler{
            keys_down: [false; KEYBOARD_SIZE],
            keys_pressed: [false; KEYBOARD_SIZE],
            keys_released: [false; KEYBOARD_SIZE],
            mouse_down: [false; MOUSE_SIZE],
            mouse_pressed: [false; MOUSE_SIZE],
            mouse_released: [false; MOUSE_SIZE],
            mouse_dragged: [false; MOUSE_SIZE],
            mouse_press_pixel_coord: [Vec2::ZERO; MOUSE_SIZE],
            mouse_drag_pixel_origin: [Vec2::ZERO; MOUSE_SIZE],
            curr_mouse_pixel_coord: Vec2::ZERO,
            prev_mouse_pixel_coord: Vec2::ZERO,
            curr_mouse_pixel_motion: Vec2::ZERO,
            prev_mouse_pixel_motion: Vec2::ZERO,
            scroll_amount: Vec2::ZERO,
            window_size: Vec2::ONE,
        }
    }

    pub fn update(&mut self) {
        for i in 0..KEYBOARD_SIZE {
            self.keys_pressed[i] = false;
            self.keys_released[i] = false;
        }

        for i in 0..MOUSE_SIZE {
            self.mouse_pressed[i] = false;
            self.mouse_released[i] = false;
        }

        self.prev_mouse_pixel_coord = self.curr_mouse_pixel_coord;
        self.prev_mouse_pixel_motion = self.curr_mouse_pixel_motion;
        self.curr_mouse_pixel_motion = Vec2::ZERO;

        self.scroll_amount = Vec2::ZERO;
    }

    pub fn process_event(&mut self, event: &Event) {

        match event {
            Event::KeyDown { scancode, keycode, .. } => {
                let key = if USE_SCANCODES { scancode.unwrap() as usize } else { keycode.unwrap() as usize };
                if key < KEYBOARD_SIZE {
                    self.keys_down[key] = true;
                    self.keys_pressed[key] = true;
                } else {
                    warn!("Received SDL input event with unknown keycode {key} - event: {event:?}");
                }
            }
            Event::KeyUp { scancode, keycode, .. } => {
                let key = if USE_SCANCODES { scancode.unwrap() as usize } else { keycode.unwrap() as usize };
                if key < KEYBOARD_SIZE {
                    self.keys_down[key] = false;
                    self.keys_released[key] = true;
                } else {
                    warn!("Received SDL input event with unknown keycode {key} - event: {event:?}");
                }
            }
            Event::MouseButtonDown { mouse_btn, x, y, .. } => {
                let btn = (*mouse_btn) as usize;
                if btn < MOUSE_SIZE {
                    self.mouse_down[btn] = true;
                    self.mouse_pressed[btn] = true;
                    self.mouse_press_pixel_coord[btn] = Vec2::new(*x, *y);
                    self.mouse_drag_pixel_origin[btn] = Vec2::new(*x, *y);
                    self.mouse_dragged[btn] = false;
                } else {
                    warn!("Received SDL input event with unknown mouse button {btn} - event: {event:?}");
                }
            }
            Event::MouseButtonUp { mouse_btn, x: _x, y: _y, .. } => {
                let btn = (*mouse_btn) as usize;
                if btn < MOUSE_SIZE {
                    self.mouse_down[btn] = false;
                    self.mouse_released[btn] = true;
                    self.mouse_dragged[btn] = false;
                } else {
                    warn!("Received SDL input event with unknown mouse button {btn} - event: {event:?}");
                }
            }
            Event::MouseMotion { x, y, xrel, yrel, .. } => {
                self.curr_mouse_pixel_coord = Vec2::new(*x, *y); // why is this not reset to zero?
                self.curr_mouse_pixel_motion = Vec2::new(*xrel, *yrel);
                for i in 0..MOUSE_SIZE {
                    self.mouse_dragged[i] = self.mouse_down[i];
                }
            }
            Event::MouseWheel { direction, x, y, mouse_x: _mouse_x, mouse_y: _mouse_y, .. } => {
                let scale = if *direction == MouseWheelDirection::Flipped { -1.0 } else { 1.0 };
                self.scroll_amount.x = *x * scale;
                self.scroll_amount.y = *y * scale;
            }

            _ => {}
        }
    }

    pub fn key_down(&self, key: Key) -> bool {
        self.keys_down[key as usize]
    }

    pub fn key_pressed(&self, key: Key) -> bool {
        self.keys_pressed[key as usize]
    }

    pub fn key_released(&self, key: Key) -> bool {
        self.keys_released[key as usize]
    }

    pub fn mouse_down(&self, button: MouseBtn) -> bool {
        self.mouse_down[button as usize]
    }

    pub fn mouse_pressed(&self, button: MouseBtn) -> bool {
        self.mouse_pressed[button as usize]
    }

    pub fn mouse_released(&self, button: MouseBtn) -> bool {
        self.mouse_released[button as usize]
    }

    pub fn mouse_dragged(&self, button: MouseBtn) -> bool {
        self.mouse_dragged[button as usize]
    }

    pub(super) fn update_window_size(&mut self, size: Vec2) {
        self.window_size = size;
    }

    pub(super) fn sync_mouse_coord(&mut self) {
        self.prev_mouse_pixel_coord = self.curr_mouse_pixel_coord;
        self.prev_mouse_pixel_motion = Vec2::ZERO;
    }

    pub(super) fn set_mouse_pixel_coord(&mut self, coord: Vec2) {
        self.curr_mouse_pixel_coord = coord;
    }
    
    pub fn get_mouse_pixel_coord(&self) -> Vec2 {
        self.curr_mouse_pixel_coord
    }

    pub fn get_last_mouse_pixel_coord(&self) -> Vec2 {
        self.prev_mouse_pixel_coord
    }

    pub fn get_mouse_screen_coord(&self) -> Vec2 {
        self.curr_mouse_pixel_coord / self.window_size
    }

    pub fn get_last_mouse_screen_coord(&self) -> Vec2 {
        self.prev_mouse_pixel_coord / self.window_size
    }

    pub fn get_mouse_pixel_motion(&self) -> Vec2 {
        self.curr_mouse_pixel_motion
    }

    pub fn get_last_mouse_pixel_motion(&self) -> Vec2 {
        self.prev_mouse_pixel_motion
    }

    pub fn get_mouse_screen_motion(&self) -> Vec2 {
        self.curr_mouse_pixel_motion / self.window_size
    }

    pub fn get_last_mouse_screen_motion(&self) -> Vec2 {
        self.prev_mouse_pixel_motion / self.window_size
    }

    pub fn get_mouse_drag_pixel_origin(&self, button: MouseBtn) -> Vec2 {
        self.mouse_drag_pixel_origin[button as usize]
    }

    pub fn get_mouse_drag_pixel_distance(&self, button: MouseBtn) -> Vec2 {
        self.curr_mouse_pixel_coord - self.mouse_drag_pixel_origin[button as usize]
    }

    pub fn get_mouse_drag_screen_origin(&self, button: MouseBtn) -> Vec2 {
        self.mouse_drag_pixel_origin[button as usize] / self.window_size
    }

    pub fn get_mouse_drag_screen_distance(&self, button: MouseBtn) -> Vec2 {
        (self.curr_mouse_pixel_coord - self.mouse_drag_pixel_origin[button as usize]) / self.window_size 
    }

    pub fn get_mouse_scroll_amount(&self) -> Vec2 {
        self.scroll_amount
    }
}