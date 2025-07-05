extern crate sdl3;

use crate::application::InputHandler;
use anyhow::Result;
use glam::Vec2;
use log::{error, info};
use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::KeyboardUtil;
use sdl3::mouse::MouseUtil;
use sdl3::pixels::Color;
use sdl3::render::WindowCanvas;
use sdl3::{IntegerOrSdlError, Sdl, VideoSubsystem};
use std::ffi::NulError;

pub type SdlWindow = sdl3::video::Window;

pub struct Window {
    sdl_ctx: Sdl,
    _sdl_video: VideoSubsystem,
    sdl_canvas: WindowCanvas,
    input_handler: InputHandler,
    did_quit: bool,
    mouse_grabbed: bool,
    did_warp_mouse: bool,
    did_resize: bool,
    is_visible: bool,
}

impl Window {
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
        info!("Initializing window");

        info!("Initializing SDL");
        let sdl_ctx = sdl3::init()
            .inspect_err(|_| error!("Failed to initialize SDL3 library"))?;

        let sdl_video = sdl_ctx.video()
            .inspect_err(|_| error!("Failed to initialize SDL3 video subsystem"))?;

        let sdl_window = sdl_video.window(title, width, height)
            .position_centered()
            .resizable()
            // .hidden()
            .vulkan()
            .build()
            .inspect_err(|_| error!("Failed to create SDL3 window"))?;
        
        let sdl_canvas = sdl_window.into_canvas();

        let input_handler = InputHandler::new();

        let mut window = Window{
            sdl_ctx,
            _sdl_video: sdl_video,
            sdl_canvas,
            input_handler,
            did_quit: false,
            mouse_grabbed: false,
            did_warp_mouse: false,
            did_resize: false,
            is_visible: false,
        };


        window.input_handler.update_window_size(window.get_window_size());
        
        Ok(window)
    }

    pub fn update(&mut self) {
        self.sdl_canvas.set_draw_color(Color::RGB(0, 255, 255));
        self.sdl_canvas.clear();
        self.sdl_canvas.present();

        let mut event_pump = self.sdl_ctx.event_pump().unwrap();

        self.did_resize = false;
        
        self.sdl_canvas.clear();
        self.input_handler.update();

        let is_window_focused = true; // Application::instance()->isWindowFocused()
        if self.mouse_grabbed && is_window_focused {
            self.set_mouse_screen_coord(Vec2::new(0.5, 0.5));
        }
        
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    info!("Window quit");
                    self.did_quit = true;
                },
                Event::Window { win_event, .. } => {
                    match win_event {
                        WindowEvent::Shown => {
                            self.is_visible = true;
                        }
                        WindowEvent::Hidden => {
                            self.is_visible = false;
                        }
                        WindowEvent::Resized(_width, _height) => {
                            self.input_handler.update_window_size(self.get_window_size());
                            self.did_resize = true;
                        }
                        WindowEvent::PixelSizeChanged(_, _) => {}
                        WindowEvent::Minimized => {}
                        WindowEvent::Maximized => {}
                        WindowEvent::Restored => {}
                        WindowEvent::MouseEnter => {}
                        WindowEvent::MouseLeave => {}
                        WindowEvent::FocusGained => {}
                        WindowEvent::FocusLost => {}
                        WindowEvent::CloseRequested => {}
                        _ => {}
                    }
                }
                _ => {}
            }

            self.input_handler.process_event(&event);
        }
        self.sdl_canvas.present();
    }

    pub fn sdl_window_handle(&self) -> &SdlWindow {
        self.sdl_canvas.window()
    }

    pub fn sdl_canvas_handle(&self) -> &WindowCanvas {
        &self.sdl_canvas
    }
    
    pub fn did_quit(&self) -> bool {
        self.did_quit
    }

    pub fn did_resize(&self) -> bool {
        self.did_resize
    }
    
    pub fn is_visible(&self) -> bool {
        self.is_visible
    }
    
    pub fn set_visible(&mut self, visible: bool) -> bool {
        if visible {
            self.window_mut().show()
        } else {
            self.window_mut().hide()
        }
    }
    
    pub fn get_title(&self) -> &str {
        self.window().title()
    }
    
    pub fn set_title(&mut self, title: &str) -> Result<(), NulError> {
        self.window_mut().set_title(title)
    }
    
    pub fn mouse(&self) -> MouseUtil {
        self.sdl_ctx.mouse()
    }

    pub fn keyboard(&self) -> KeyboardUtil {
        self.sdl_ctx.keyboard()
    }

    pub fn canvas(&self) -> &WindowCanvas {
        &self.sdl_canvas
    }

    pub fn window(&self) -> &sdl3::video::Window {
        self.sdl_canvas.window()
    }

    pub fn window_mut(&mut self) -> &mut sdl3::video::Window {
        self.sdl_canvas.window_mut()
    }
    
    pub fn input(&self) -> &InputHandler {
        &self.input_handler
    }
    
    pub fn input_mut(&mut self) -> &mut InputHandler {
        &mut self.input_handler
    }
    
    pub fn is_mouse_grabbed(&self) -> bool {
        self.mouse_grabbed
    }
    
    pub fn set_mouse_grabbed(&mut self, grabbed: bool) {
        if grabbed != self.is_mouse_grabbed() {
            self.mouse_grabbed = grabbed;
            let mouse = self.mouse();
            mouse.show_cursor(!grabbed);
            mouse.set_relative_mouse_mode(self.window(), grabbed);
        }
    }
    
    pub fn get_window_size(&self) -> Vec2 {
        let (width, height) = self.window().size();
        Vec2::new(width as f32, height as f32)
    }
    
    pub fn get_window_size_in_pixels(&self) -> Vec2 {
        let (width, height) = self.window().size_in_pixels();
        Vec2::new(width as f32, height as f32)
    }
    
    pub fn set_window_size(&mut self, size: Vec2) -> Result<(), IntegerOrSdlError> {
        self.window_mut().set_size(size.x as u32, size.y as u32)
    }

    pub fn set_mouse_pixel_coord(&mut self, coord: Vec2) -> bool {
        let size = self.get_window_size();
        if coord.x < 0.0 || coord.x >= size.x || coord.y < 0.0 || coord.y >= size.y {
            return false;
        }

        let window = self.window();
        let mouse = self.mouse();
        mouse.warp_mouse_in_window(window, coord.x, coord.y);
        self.input_handler.set_mouse_pixel_coord(coord);
        self.input_handler.sync_mouse_coord();
        self.did_warp_mouse = true;
        
        true
    }
    
    pub fn set_mouse_screen_coord(&mut self, coord: Vec2) -> bool {
        let size = self.get_window_size();
        self.set_mouse_pixel_coord(coord * size)
    }
}