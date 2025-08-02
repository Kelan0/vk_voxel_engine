use vulkano::VulkanObject;
use anyhow::Result;
use ash::vk::Extent2D;
use log::{debug, info, warn};
use shrev::ReaderId;
use crate::application::{InputHandler, Key, Ticker};
use crate::core::{CommandBuffer, Engine, FrameCompleteEvent, GraphicsManager, RecreateSwapchainEvent};
use crate::{function_name, profile_scope_fn};

pub struct GUIRenderer {
    ctx: egui::Context,
    raw_input: egui::RawInput,
    renderer: Option<egui_ash_renderer::Renderer>,
    resources: Vec<FrameResource>,
    event_recreate_swapchain: Option<ReaderId<RecreateSwapchainEvent>>,
    event_frame_complete: Option<ReaderId<FrameCompleteEvent>>,
}

struct FrameResource {
    free: Vec<egui::TextureId>
}
impl GUIRenderer {
    pub fn new(graphics: &GraphicsManager) -> Result<Self> {
        let ctx = egui::Context::default();

        let raw_input = egui::RawInput{
            viewport_id: egui::ViewportId::ROOT,
            viewports: std::iter::once((egui::ViewportId::ROOT, egui::ViewportInfo {
                native_pixels_per_point: Some(1.0),
                focused: Some(true),
                title: Some(String::from("TEST VIEWPORT")),
                ..Default::default()
            })).collect(),
            focused: true,
            system_theme: None,
            ..Default::default()
        };

        Ok(GUIRenderer {
            ctx,
            raw_input,
            renderer: None,
            resources: vec![],
            event_recreate_swapchain: None,
            event_frame_complete: None,
        })
    }

    pub fn register_events(&mut self, engine: &mut Engine) -> Result<()> {
        self.event_recreate_swapchain = Some(engine.graphics.event_bus().register::<RecreateSwapchainEvent>());
        self.event_frame_complete = Some(engine.graphics.event_bus().register::<FrameCompleteEvent>());
        Ok(())
    }

    pub fn init(&mut self, graphics: &GraphicsManager) -> Result<()> {
        let max_texture_side = graphics.device_properties().max_image_dimension2_d as usize;
        self.raw_input.max_texture_side = Some(max_texture_side);
        Ok(())
    }

    fn init_resources(&mut self, engine: &mut Engine) -> Result<()> {

        // TODO: is this a memory leak if anything was in the 'free' buffers at this point? hmm...

        self.resources.resize_with(engine.graphics.max_concurrent_frames(), || {
            FrameResource {
                free: vec![],
            }
        });

        Ok(())
    }

    fn on_recreate_swapchain(&mut self, engine: &mut Engine) -> Result<()> {
        let graphics = &engine.graphics;

        if let Some(renderer) = &mut self.renderer {
            renderer.set_render_pass(graphics.render_pass().handle())?;

        } else {
            // Only initialize the renderer fully first time round...
            // Just update necessary stuff after that.
            let renderer = egui_ash_renderer::Renderer::with_default_allocator(
                graphics.ash_instance(),
                graphics.physical_device().handle(),
                graphics.ash_device().clone(),
                graphics.render_pass().handle(),
                egui_ash_renderer::Options {
                    in_flight_frames: graphics.max_concurrent_frames(),
                    enable_depth_test: false,
                    enable_depth_write: false,
                    srgb_framebuffer: false,
                    ..Default::default()
                }
            )?;
            self.renderer = Some(renderer);
        }


        self.init_resources(engine)?;

        Ok(())
    }

    fn on_frame_complete(&mut self, engine: &mut Engine, event: FrameCompleteEvent) -> Result<()> {

        let renderer = self.renderer.as_mut().expect("UI Renderer is not initialized");
        let resource = &mut self.resources[event.frame_index];

        renderer.free_textures(resource.free.as_slice())?;
        resource.free.clear();

        Ok(())
    }

    pub fn render_gui(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer, resolution: [u32; 2], mut run_ui: impl FnMut(&mut Ticker, &mut Engine, &egui::Context)) -> Result<()> {

        profile_scope_fn!(&engine.frame_profiler);

        if engine.graphics.event_bus().has_any_opt(&mut self.event_recreate_swapchain) {
            self.on_recreate_swapchain(engine)?;
        }
        if let Some(event) = engine.graphics.event_bus().read_one_opt(&mut self.event_frame_complete) {
            self.on_frame_complete(engine, event)?;
        }

        self.raw_input.time = Some(ticker.simulation_time());
        self.raw_input.predicted_dt = ticker.delta_time() as f32;
        self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(
                resolution[0] as f32 / self.ctx.pixels_per_point(),
                resolution[1] as f32 / self.ctx.pixels_per_point()
            )
        ));

        // if self.raw_input.events.len() > 0 {
        //     debug!("{} gui raw input events:\n{:?}", self.raw_input.events.len(), self.raw_input.events);
        // }

        let output = self.ctx.run(self.raw_input.take(), |ctx| {
            run_ui(ticker, engine, ctx);
        });

        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);

        let vk_cmd_buf = *cmd_buf.handle();
        let queue = engine.graphics.graphics_queue().handle();
        let cmd_pool = engine.graphics.get_graphics_command_pool().clone();

        let renderer = self.renderer.as_mut().expect("UI Renderer is not initialized");
        let resource = &mut self.resources[engine.graphics.current_frame_index()];

        let extent = Extent2D{ width: resolution[0], height: resolution[1] };

        if !output.textures_delta.set.is_empty() || !output.textures_delta.free.is_empty() {
            info!("textures_delta: {} set, {} freed", output.textures_delta.set.len(), output.textures_delta.free.len());
        }

        renderer.set_textures(queue, cmd_pool, output.textures_delta.set.as_slice())?;
        renderer.cmd_draw(vk_cmd_buf, extent, output.pixels_per_point, &primitives)?;

        resource.free = output.textures_delta.free;

        Ok(())
    }

    pub fn process_event(&mut self, event: &sdl3::event::Event, input_handler: &InputHandler) {

        let get_modifiers = || {
            let mut modifiers = egui::Modifiers::NONE;
            if input_handler.key_down(Key::LAlt) || input_handler.key_down(Key::RAlt) {
                modifiers |= egui::Modifiers::ALT;
            }
            if input_handler.key_down(Key::LCtrl) || input_handler.key_down(Key::RCtrl) {
                modifiers |= egui::Modifiers::CTRL | egui::Modifiers::COMMAND;
            }
            if input_handler.key_down(Key::LShift) || input_handler.key_down(Key::RShift) {
                modifiers |= egui::Modifiers::SHIFT;
            }
            if input_handler.key_down(Key::LGui) || input_handler.key_down(Key::RGui) {
                modifiers |= egui::Modifiers::MAC_CMD | egui::Modifiers::COMMAND;
            }
            modifiers
        };

        let get_mouse_btn = |mouse_btn| {
            match mouse_btn {
                sdl3::mouse::MouseButton::Left => egui::PointerButton::Primary,
                sdl3::mouse::MouseButton::Middle => egui::PointerButton::Middle,
                sdl3::mouse::MouseButton::Right => egui::PointerButton::Secondary,
                sdl3::mouse::MouseButton::X1 => egui::PointerButton::Extra1,
                sdl3::mouse::MouseButton::X2 => egui::PointerButton::Extra2,
                _ => unreachable!()
            }
        };

        let get_key = |keycode: sdl3::keyboard::Keycode| -> Option<egui::Key> {
            match keycode {
                sdl3::keyboard::Keycode::ScancodeMask => Some(egui::Key::Escape),
                sdl3::keyboard::Keycode::Unknown => Some(egui::Key::Escape),
                sdl3::keyboard::Keycode::Return => Some(egui::Key::Enter),
                sdl3::keyboard::Keycode::Escape => Some(egui::Key::Escape),
                sdl3::keyboard::Keycode::Backspace => Some(egui::Key::Backspace),
                sdl3::keyboard::Keycode::Tab => Some(egui::Key::Tab),
                sdl3::keyboard::Keycode::Space => Some(egui::Key::Space),
                sdl3::keyboard::Keycode::Exclaim => Some(egui::Key::Exclamationmark),
                sdl3::keyboard::Keycode::DblApostrophe => Some(egui::Key::Quote),
                sdl3::keyboard::Keycode::Apostrophe => Some(egui::Key::Quote),
                sdl3::keyboard::Keycode::LeftParen => Some(egui::Key::OpenBracket),
                sdl3::keyboard::Keycode::RightParen => Some(egui::Key::CloseBracket),
                sdl3::keyboard::Keycode::Plus => Some(egui::Key::Plus),
                sdl3::keyboard::Keycode::Comma => Some(egui::Key::Comma),
                sdl3::keyboard::Keycode::Minus => Some(egui::Key::Minus),
                sdl3::keyboard::Keycode::Period => Some(egui::Key::Period),
                sdl3::keyboard::Keycode::Slash => Some(egui::Key::Slash),
                sdl3::keyboard::Keycode::_0 => Some(egui::Key::Num0),
                sdl3::keyboard::Keycode::_1 => Some(egui::Key::Num1),
                sdl3::keyboard::Keycode::_2 => Some(egui::Key::Num2),
                sdl3::keyboard::Keycode::_3 => Some(egui::Key::Num3),
                sdl3::keyboard::Keycode::_4 => Some(egui::Key::Num4),
                sdl3::keyboard::Keycode::_5 => Some(egui::Key::Num5),
                sdl3::keyboard::Keycode::_6 => Some(egui::Key::Num6),
                sdl3::keyboard::Keycode::_7 => Some(egui::Key::Num7),
                sdl3::keyboard::Keycode::_8 => Some(egui::Key::Num8),
                sdl3::keyboard::Keycode::_9 => Some(egui::Key::Num9),
                sdl3::keyboard::Keycode::Colon => Some(egui::Key::Colon),
                sdl3::keyboard::Keycode::Semicolon => Some(egui::Key::Semicolon),
                sdl3::keyboard::Keycode::Equals => Some(egui::Key::Equals),
                sdl3::keyboard::Keycode::Question => Some(egui::Key::Questionmark),
                sdl3::keyboard::Keycode::LeftBracket => Some(egui::Key::OpenCurlyBracket),
                sdl3::keyboard::Keycode::Backslash => Some(egui::Key::Backslash),
                sdl3::keyboard::Keycode::RightBracket => Some(egui::Key::CloseCurlyBracket),
                sdl3::keyboard::Keycode::Grave => Some(egui::Key::Backtick),
                sdl3::keyboard::Keycode::A => Some(egui::Key::A),
                sdl3::keyboard::Keycode::B => Some(egui::Key::B),
                sdl3::keyboard::Keycode::C => Some(egui::Key::C),
                sdl3::keyboard::Keycode::D => Some(egui::Key::D),
                sdl3::keyboard::Keycode::E => Some(egui::Key::E),
                sdl3::keyboard::Keycode::F => Some(egui::Key::F),
                sdl3::keyboard::Keycode::G => Some(egui::Key::G),
                sdl3::keyboard::Keycode::H => Some(egui::Key::H),
                sdl3::keyboard::Keycode::I => Some(egui::Key::I),
                sdl3::keyboard::Keycode::J => Some(egui::Key::J),
                sdl3::keyboard::Keycode::K => Some(egui::Key::K),
                sdl3::keyboard::Keycode::L => Some(egui::Key::L),
                sdl3::keyboard::Keycode::M => Some(egui::Key::M),
                sdl3::keyboard::Keycode::N => Some(egui::Key::N),
                sdl3::keyboard::Keycode::O => Some(egui::Key::O),
                sdl3::keyboard::Keycode::P => Some(egui::Key::P),
                sdl3::keyboard::Keycode::Q => Some(egui::Key::Q),
                sdl3::keyboard::Keycode::R => Some(egui::Key::R),
                sdl3::keyboard::Keycode::S => Some(egui::Key::S),
                sdl3::keyboard::Keycode::T => Some(egui::Key::T),
                sdl3::keyboard::Keycode::U => Some(egui::Key::U),
                sdl3::keyboard::Keycode::V => Some(egui::Key::V),
                sdl3::keyboard::Keycode::W => Some(egui::Key::W),
                sdl3::keyboard::Keycode::X => Some(egui::Key::X),
                sdl3::keyboard::Keycode::Y => Some(egui::Key::Y),
                sdl3::keyboard::Keycode::Z => Some(egui::Key::Z),
                sdl3::keyboard::Keycode::LeftBrace => Some(egui::Key::OpenCurlyBracket),
                sdl3::keyboard::Keycode::Pipe => Some(egui::Key::Pipe),
                sdl3::keyboard::Keycode::RightBrace => Some(egui::Key::CloseCurlyBracket),
                sdl3::keyboard::Keycode::Delete => Some(egui::Key::Delete),
                sdl3::keyboard::Keycode::F1 => Some(egui::Key::F1),
                sdl3::keyboard::Keycode::F2 => Some(egui::Key::F2),
                sdl3::keyboard::Keycode::F3 => Some(egui::Key::F3),
                sdl3::keyboard::Keycode::F4 => Some(egui::Key::F4),
                sdl3::keyboard::Keycode::F5 => Some(egui::Key::F5),
                sdl3::keyboard::Keycode::F6 => Some(egui::Key::F6),
                sdl3::keyboard::Keycode::F7 => Some(egui::Key::F7),
                sdl3::keyboard::Keycode::F8 => Some(egui::Key::F8),
                sdl3::keyboard::Keycode::F9 => Some(egui::Key::F9),
                sdl3::keyboard::Keycode::F10 => Some(egui::Key::F10),
                sdl3::keyboard::Keycode::F11 => Some(egui::Key::F11),
                sdl3::keyboard::Keycode::F12 => Some(egui::Key::F12),
                sdl3::keyboard::Keycode::Insert => Some(egui::Key::Insert),
                sdl3::keyboard::Keycode::Home => Some(egui::Key::Home),
                sdl3::keyboard::Keycode::PageUp => Some(egui::Key::PageUp),
                sdl3::keyboard::Keycode::End => Some(egui::Key::End),
                sdl3::keyboard::Keycode::PageDown => Some(egui::Key::PageDown),
                sdl3::keyboard::Keycode::Right => Some(egui::Key::ArrowRight),
                sdl3::keyboard::Keycode::Left => Some(egui::Key::ArrowLeft),
                sdl3::keyboard::Keycode::Down => Some(egui::Key::ArrowDown),
                sdl3::keyboard::Keycode::Up => Some(egui::Key::ArrowUp),
                sdl3::keyboard::Keycode::KpMinus => Some(egui::Key::Minus),
                sdl3::keyboard::Keycode::KpPlus => Some(egui::Key::Plus),
                sdl3::keyboard::Keycode::KpEnter => Some(egui::Key::Enter),
                sdl3::keyboard::Keycode::Kp1 => Some(egui::Key::Num0),
                sdl3::keyboard::Keycode::Kp2 => Some(egui::Key::Num1),
                sdl3::keyboard::Keycode::Kp3 => Some(egui::Key::Num2),
                sdl3::keyboard::Keycode::Kp4 => Some(egui::Key::Num3),
                sdl3::keyboard::Keycode::Kp5 => Some(egui::Key::Num4),
                sdl3::keyboard::Keycode::Kp6 => Some(egui::Key::Num5),
                sdl3::keyboard::Keycode::Kp7 => Some(egui::Key::Num6),
                sdl3::keyboard::Keycode::Kp8 => Some(egui::Key::Num7),
                sdl3::keyboard::Keycode::Kp9 => Some(egui::Key::Num8),
                sdl3::keyboard::Keycode::Kp0 => Some(egui::Key::Num9),
                sdl3::keyboard::Keycode::KpPeriod => Some(egui::Key::Period),
                sdl3::keyboard::Keycode::KpEquals => Some(egui::Key::Equals),
                sdl3::keyboard::Keycode::F13 => Some(egui::Key::F13),
                sdl3::keyboard::Keycode::F14 => Some(egui::Key::F14),
                sdl3::keyboard::Keycode::F15 => Some(egui::Key::F15),
                sdl3::keyboard::Keycode::F16 => Some(egui::Key::F16),
                sdl3::keyboard::Keycode::F17 => Some(egui::Key::F17),
                sdl3::keyboard::Keycode::F18 => Some(egui::Key::F18),
                sdl3::keyboard::Keycode::F19 => Some(egui::Key::F19),
                sdl3::keyboard::Keycode::F20 => Some(egui::Key::F20),
                sdl3::keyboard::Keycode::F21 => Some(egui::Key::F21),
                sdl3::keyboard::Keycode::F22 => Some(egui::Key::F22),
                sdl3::keyboard::Keycode::F23 => Some(egui::Key::F23),
                sdl3::keyboard::Keycode::F24 => Some(egui::Key::F24),
                sdl3::keyboard::Keycode::Cut => Some(egui::Key::Cut),
                sdl3::keyboard::Keycode::Copy => Some(egui::Key::Copy),
                sdl3::keyboard::Keycode::Paste => Some(egui::Key::Paste),
                sdl3::keyboard::Keycode::KpComma => Some(egui::Key::Comma),
                sdl3::keyboard::Keycode::Return2 => Some(egui::Key::Enter),
                sdl3::keyboard::Keycode::KpLeftParen => Some(egui::Key::OpenBracket),
                sdl3::keyboard::Keycode::KpRightParen => Some(egui::Key::CloseBracket),
                sdl3::keyboard::Keycode::KpLeftBrace => Some(egui::Key::OpenCurlyBracket),
                sdl3::keyboard::Keycode::KpRightBrace => Some(egui::Key::CloseCurlyBracket),
                sdl3::keyboard::Keycode::KpTab => Some(egui::Key::Tab),
                sdl3::keyboard::Keycode::KpBackspace => Some(egui::Key::Backspace),
                sdl3::keyboard::Keycode::KpA => Some(egui::Key::A),
                sdl3::keyboard::Keycode::KpB => Some(egui::Key::B),
                sdl3::keyboard::Keycode::KpC => Some(egui::Key::C),
                sdl3::keyboard::Keycode::KpD => Some(egui::Key::D),
                sdl3::keyboard::Keycode::KpE => Some(egui::Key::E),
                sdl3::keyboard::Keycode::KpF => Some(egui::Key::F),
                sdl3::keyboard::Keycode::KpVerticalBar => Some(egui::Key::Pipe),
                sdl3::keyboard::Keycode::KpColon => Some(egui::Key::Colon),
                sdl3::keyboard::Keycode::KpExclam => Some(egui::Key::Exclamationmark),
                sdl3::keyboard::Keycode::AcHome => Some(egui::Key::Home),
                sdl3::keyboard::Keycode::AcBack => Some(egui::Key::BrowserBack),
                _ => None
            }
        };

        let get_physical_key = |scancode: &Option<sdl3::keyboard::Scancode>| -> Option<egui::Key> {
            if let Some(scancode) = scancode {
                return match scancode {
                    sdl3::keyboard::Scancode::A => Some(egui::Key::A),
                    sdl3::keyboard::Scancode::B => Some(egui::Key::B),
                    sdl3::keyboard::Scancode::C => Some(egui::Key::C),
                    sdl3::keyboard::Scancode::D => Some(egui::Key::D),
                    sdl3::keyboard::Scancode::E => Some(egui::Key::E),
                    sdl3::keyboard::Scancode::F => Some(egui::Key::F),
                    sdl3::keyboard::Scancode::G => Some(egui::Key::G),
                    sdl3::keyboard::Scancode::H => Some(egui::Key::H),
                    sdl3::keyboard::Scancode::I => Some(egui::Key::I),
                    sdl3::keyboard::Scancode::J => Some(egui::Key::J),
                    sdl3::keyboard::Scancode::K => Some(egui::Key::K),
                    sdl3::keyboard::Scancode::L => Some(egui::Key::L),
                    sdl3::keyboard::Scancode::M => Some(egui::Key::M),
                    sdl3::keyboard::Scancode::N => Some(egui::Key::N),
                    sdl3::keyboard::Scancode::O => Some(egui::Key::O),
                    sdl3::keyboard::Scancode::P => Some(egui::Key::P),
                    sdl3::keyboard::Scancode::Q => Some(egui::Key::Q),
                    sdl3::keyboard::Scancode::R => Some(egui::Key::R),
                    sdl3::keyboard::Scancode::S => Some(egui::Key::S),
                    sdl3::keyboard::Scancode::T => Some(egui::Key::T),
                    sdl3::keyboard::Scancode::U => Some(egui::Key::U),
                    sdl3::keyboard::Scancode::V => Some(egui::Key::V),
                    sdl3::keyboard::Scancode::W => Some(egui::Key::W),
                    sdl3::keyboard::Scancode::X => Some(egui::Key::X),
                    sdl3::keyboard::Scancode::Y => Some(egui::Key::Y),
                    sdl3::keyboard::Scancode::Z => Some(egui::Key::Z),
                    sdl3::keyboard::Scancode::_1 => Some(egui::Key::Num1),
                    sdl3::keyboard::Scancode::_2 => Some(egui::Key::Num2),
                    sdl3::keyboard::Scancode::_3 => Some(egui::Key::Num3),
                    sdl3::keyboard::Scancode::_4 => Some(egui::Key::Num4),
                    sdl3::keyboard::Scancode::_5 => Some(egui::Key::Num5),
                    sdl3::keyboard::Scancode::_6 => Some(egui::Key::Num6),
                    sdl3::keyboard::Scancode::_7 => Some(egui::Key::Num7),
                    sdl3::keyboard::Scancode::_8 => Some(egui::Key::Num8),
                    sdl3::keyboard::Scancode::_9 => Some(egui::Key::Num9),
                    sdl3::keyboard::Scancode::_0 => Some(egui::Key::Num0),
                    sdl3::keyboard::Scancode::Return => Some(egui::Key::Enter),
                    sdl3::keyboard::Scancode::Escape => Some(egui::Key::Escape),
                    sdl3::keyboard::Scancode::Backspace => Some(egui::Key::Backspace),
                    sdl3::keyboard::Scancode::Tab => Some(egui::Key::Tab),
                    sdl3::keyboard::Scancode::Space => Some(egui::Key::Space),
                    sdl3::keyboard::Scancode::Minus => Some(egui::Key::Minus),
                    sdl3::keyboard::Scancode::Equals => Some(egui::Key::Equals),
                    sdl3::keyboard::Scancode::LeftBracket => Some(egui::Key::OpenBracket),
                    sdl3::keyboard::Scancode::RightBracket => Some(egui::Key::CloseBracket),
                    sdl3::keyboard::Scancode::Backslash => Some(egui::Key::Backslash),
                    sdl3::keyboard::Scancode::Semicolon => Some(egui::Key::Semicolon),
                    sdl3::keyboard::Scancode::Apostrophe => Some(egui::Key::Quote),
                    sdl3::keyboard::Scancode::Grave => Some(egui::Key::Backtick),
                    sdl3::keyboard::Scancode::Comma => Some(egui::Key::Comma),
                    sdl3::keyboard::Scancode::Period => Some(egui::Key::Period),
                    sdl3::keyboard::Scancode::Slash => Some(egui::Key::Slash),
                    sdl3::keyboard::Scancode::F1 => Some(egui::Key::F1),
                    sdl3::keyboard::Scancode::F2 => Some(egui::Key::F2),
                    sdl3::keyboard::Scancode::F3 => Some(egui::Key::F3),
                    sdl3::keyboard::Scancode::F4 => Some(egui::Key::F4),
                    sdl3::keyboard::Scancode::F5 => Some(egui::Key::F5),
                    sdl3::keyboard::Scancode::F6 => Some(egui::Key::F6),
                    sdl3::keyboard::Scancode::F7 => Some(egui::Key::F7),
                    sdl3::keyboard::Scancode::F8 => Some(egui::Key::F8),
                    sdl3::keyboard::Scancode::F9 => Some(egui::Key::F9),
                    sdl3::keyboard::Scancode::F10 => Some(egui::Key::F10),
                    sdl3::keyboard::Scancode::F11 => Some(egui::Key::F11),
                    sdl3::keyboard::Scancode::F12 => Some(egui::Key::F12),
                    sdl3::keyboard::Scancode::Insert => Some(egui::Key::Insert),
                    sdl3::keyboard::Scancode::Home => Some(egui::Key::Home),
                    sdl3::keyboard::Scancode::PageUp => Some(egui::Key::PageUp),
                    sdl3::keyboard::Scancode::Delete => Some(egui::Key::Delete),
                    sdl3::keyboard::Scancode::End => Some(egui::Key::End),
                    sdl3::keyboard::Scancode::PageDown => Some(egui::Key::PageDown),
                    sdl3::keyboard::Scancode::Right => Some(egui::Key::ArrowRight),
                    sdl3::keyboard::Scancode::Left => Some(egui::Key::ArrowLeft),
                    sdl3::keyboard::Scancode::Down => Some(egui::Key::ArrowDown),
                    sdl3::keyboard::Scancode::Up => Some(egui::Key::ArrowUp),
                    sdl3::keyboard::Scancode::KpDivide => Some(egui::Key::Slash),
                    sdl3::keyboard::Scancode::KpMinus => Some(egui::Key::Minus),
                    sdl3::keyboard::Scancode::KpPlus => Some(egui::Key::Plus),
                    sdl3::keyboard::Scancode::KpEnter => Some(egui::Key::Enter),
                    sdl3::keyboard::Scancode::Kp1 => Some(egui::Key::Num1),
                    sdl3::keyboard::Scancode::Kp2 => Some(egui::Key::Num2),
                    sdl3::keyboard::Scancode::Kp3 => Some(egui::Key::Num3),
                    sdl3::keyboard::Scancode::Kp4 => Some(egui::Key::Num4),
                    sdl3::keyboard::Scancode::Kp5 => Some(egui::Key::Num5),
                    sdl3::keyboard::Scancode::Kp6 => Some(egui::Key::Num6),
                    sdl3::keyboard::Scancode::Kp7 => Some(egui::Key::Num7),
                    sdl3::keyboard::Scancode::Kp8 => Some(egui::Key::Num8),
                    sdl3::keyboard::Scancode::Kp9 => Some(egui::Key::Num9),
                    sdl3::keyboard::Scancode::Kp0 => Some(egui::Key::Num0),
                    sdl3::keyboard::Scancode::KpPeriod => Some(egui::Key::Period),
                    sdl3::keyboard::Scancode::NonUsBackslash => Some(egui::Key::Backslash),
                    sdl3::keyboard::Scancode::KpEquals => Some(egui::Key::Equals),
                    sdl3::keyboard::Scancode::F13 => Some(egui::Key::F13),
                    sdl3::keyboard::Scancode::F14 => Some(egui::Key::F14),
                    sdl3::keyboard::Scancode::F15 => Some(egui::Key::F15),
                    sdl3::keyboard::Scancode::F16 => Some(egui::Key::F16),
                    sdl3::keyboard::Scancode::F17 => Some(egui::Key::F17),
                    sdl3::keyboard::Scancode::F18 => Some(egui::Key::F18),
                    sdl3::keyboard::Scancode::F19 => Some(egui::Key::F19),
                    sdl3::keyboard::Scancode::F20 => Some(egui::Key::F20),
                    sdl3::keyboard::Scancode::F21 => Some(egui::Key::F21),
                    sdl3::keyboard::Scancode::F22 => Some(egui::Key::F22),
                    sdl3::keyboard::Scancode::F23 => Some(egui::Key::F23),
                    sdl3::keyboard::Scancode::F24 => Some(egui::Key::F24),
                    sdl3::keyboard::Scancode::Cut => Some(egui::Key::Cut),
                    sdl3::keyboard::Scancode::Copy => Some(egui::Key::Copy),
                    sdl3::keyboard::Scancode::Paste => Some(egui::Key::Paste),
                    sdl3::keyboard::Scancode::KpComma => Some(egui::Key::Comma),
                    sdl3::keyboard::Scancode::KpEqualsAs400 => Some(egui::Key::Equals),
                    sdl3::keyboard::Scancode::International1 => Some(egui::Key::Num1),
                    sdl3::keyboard::Scancode::International2 => Some(egui::Key::Num2),
                    sdl3::keyboard::Scancode::International3 => Some(egui::Key::Num3),
                    sdl3::keyboard::Scancode::International4 => Some(egui::Key::Num4),
                    sdl3::keyboard::Scancode::International5 => Some(egui::Key::Num5),
                    sdl3::keyboard::Scancode::International6 => Some(egui::Key::Num6),
                    sdl3::keyboard::Scancode::International7 => Some(egui::Key::Num7),
                    sdl3::keyboard::Scancode::International8 => Some(egui::Key::Num8),
                    sdl3::keyboard::Scancode::International9 => Some(egui::Key::Num9),
                    sdl3::keyboard::Scancode::Lang1 => Some(egui::Key::Num1),
                    sdl3::keyboard::Scancode::Lang2 => Some(egui::Key::Num2),
                    sdl3::keyboard::Scancode::Lang3 => Some(egui::Key::Num3),
                    sdl3::keyboard::Scancode::Lang4 => Some(egui::Key::Num4),
                    sdl3::keyboard::Scancode::Lang5 => Some(egui::Key::Num5),
                    sdl3::keyboard::Scancode::Lang6 => Some(egui::Key::Num6),
                    sdl3::keyboard::Scancode::Lang7 => Some(egui::Key::Num7),
                    sdl3::keyboard::Scancode::Lang8 => Some(egui::Key::Num8),
                    sdl3::keyboard::Scancode::Lang9 => Some(egui::Key::Num9),
                    sdl3::keyboard::Scancode::Return2 => Some(egui::Key::Enter),
                    sdl3::keyboard::Scancode::Separator => Some(egui::Key::Space),
                    sdl3::keyboard::Scancode::Kp00 => Some(egui::Key::Num0),
                    sdl3::keyboard::Scancode::Kp000 => Some(egui::Key::Num0),
                    sdl3::keyboard::Scancode::KpLeftParen => Some(egui::Key::OpenBracket),
                    sdl3::keyboard::Scancode::KpRightParen => Some(egui::Key::CloseBracket),
                    sdl3::keyboard::Scancode::KpLeftBrace => Some(egui::Key::OpenCurlyBracket),
                    sdl3::keyboard::Scancode::KpRightBrace => Some(egui::Key::CloseCurlyBracket),
                    sdl3::keyboard::Scancode::KpTab => Some(egui::Key::Tab),
                    sdl3::keyboard::Scancode::KpBackspace => Some(egui::Key::Backspace),
                    sdl3::keyboard::Scancode::KpA => Some(egui::Key::A),
                    sdl3::keyboard::Scancode::KpB => Some(egui::Key::B),
                    sdl3::keyboard::Scancode::KpC => Some(egui::Key::C),
                    sdl3::keyboard::Scancode::KpD => Some(egui::Key::D),
                    sdl3::keyboard::Scancode::KpE => Some(egui::Key::E),
                    sdl3::keyboard::Scancode::KpF => Some(egui::Key::F),
                    sdl3::keyboard::Scancode::KpVerticalBar => Some(egui::Key::Pipe),
                    sdl3::keyboard::Scancode::KpDblVerticalBar => Some(egui::Key::Pipe),
                    sdl3::keyboard::Scancode::KpColon => Some(egui::Key::Colon),
                    sdl3::keyboard::Scancode::KpSpace => Some(egui::Key::Space),
                    sdl3::keyboard::Scancode::KpExclam => Some(egui::Key::Exclamationmark),
                    sdl3::keyboard::Scancode::KpMemAdd => Some(egui::Key::Plus),
                    sdl3::keyboard::Scancode::KpMemSubtract => Some(egui::Key::Minus),
                    sdl3::keyboard::Scancode::KpMemDivide => Some(egui::Key::Slash),
                    sdl3::keyboard::Scancode::AcHome => Some(egui::Key::Home),
                    sdl3::keyboard::Scancode::AcBack => Some(egui::Key::BrowserBack),
                    _ => None
                }
            } else {
                return None;
            }
        };

        // TODO: Ime event

        match event {
            sdl3::event::Event::Window { win_event, .. } => {
                match win_event {
                    sdl3::event::WindowEvent::MouseEnter => {}
                    sdl3::event::WindowEvent::MouseLeave => {
                        self.raw_input.events.push(egui::Event::PointerGone);
                    }
                    sdl3::event::WindowEvent::FocusGained => {
                        self.raw_input.events.push(egui::Event::WindowFocused(true));
                    }
                    sdl3::event::WindowEvent::FocusLost => {
                        self.raw_input.events.push(egui::Event::WindowFocused(false));
                    }
                    _ => {}
                }
            }
            sdl3::event::Event::KeyDown { keycode, scancode, repeat, .. } => {
                if let Some(keycode) = keycode {
                    if let Some(key) = get_key(*keycode) {
                        self.raw_input.events.push(egui::Event::Key {
                            key,
                            physical_key: get_physical_key(scancode),
                            pressed: true,
                            repeat: *repeat,
                            modifiers: get_modifiers(),
                        });
                    } else {
                        warn!("Unknown input key event: Keycode: {keycode:?}, Scancode: {scancode:?}");
                    }
                }
            }
            sdl3::event::Event::KeyUp { keycode, scancode, repeat, .. } => {
                if let Some(keycode) = keycode {
                    if let Some(key) = get_key(*keycode) {
                        self.raw_input.events.push(egui::Event::Key {
                            key,
                            physical_key: get_physical_key(scancode),
                            pressed: false,
                            repeat: *repeat,
                            modifiers: get_modifiers(),
                        });
                    } else {
                        warn!("Unknown input key event: Keycode: {keycode:?}, Scancode: {scancode:?}");
                    }
                }
            }
            sdl3::event::Event::TextEditing { text, start, length, .. } => {
                debug!("SDL TextEditing event: text: \"{text}\", start: {start} len: {length}");
            }
            sdl3::event::Event::TextInput { text, .. } => {
                self.raw_input.events.push(egui::Event::Text(text.clone()));
            }
            sdl3::event::Event::MouseMotion { x, y, .. } => {
                self.raw_input.events.push(egui::Event::PointerMoved(egui::Pos2::new(*x, *y)));
            }
            sdl3::event::Event::MouseButtonDown { x, y, mouse_btn, .. } => {
                if *mouse_btn != sdl3::mouse::MouseButton::Unknown {
                    self.raw_input.events.push(egui::Event::PointerButton {
                        pos: egui::Pos2::new(*x, *y),
                        button: get_mouse_btn(*mouse_btn),
                        pressed: true,
                        modifiers: get_modifiers(),
                    });
                }
            }
            sdl3::event::Event::MouseButtonUp { x, y, mouse_btn, .. } => {
                if *mouse_btn != sdl3::mouse::MouseButton::Unknown {
                    self.raw_input.events.push(egui::Event::PointerButton {
                        pos: egui::Pos2::new(*x, *y),
                        button: get_mouse_btn(*mouse_btn),
                        pressed: false,
                        modifiers: get_modifiers(),
                    });
                }}
            sdl3::event::Event::MouseWheel { x, y, direction, .. } => {
                let scale = if *direction == sdl3::mouse::MouseWheelDirection::Flipped { -1.0 } else { 1.0 };
                self.raw_input.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::Vec2::new(*x * scale, *y * scale),
                    modifiers: get_modifiers(),
                });
            }
            // TODO: maybe these should input arrow key events?
            sdl3::event::Event::ControllerAxisMotion { .. } => {}
            sdl3::event::Event::ControllerButtonDown { .. } => {}
            sdl3::event::Event::ControllerButtonUp { .. } => {}
            sdl3::event::Event::ControllerDeviceAdded { .. } => {}
            sdl3::event::Event::ControllerDeviceRemoved { .. } => {}
            sdl3::event::Event::ControllerDeviceRemapped { .. } => {}
            sdl3::event::Event::ControllerTouchpadDown { .. } => {}
            sdl3::event::Event::ControllerTouchpadMotion { .. } => {}
            sdl3::event::Event::ControllerTouchpadUp { .. } => {}

            sdl3::event::Event::ClipboardUpdate { timestamp, .. } => {
                debug!("SDL ClipboardUpdate event: {timestamp}");
            }
            _ => {}
        }

    }
}