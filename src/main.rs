mod application;
mod core;
mod util;

use crate::application::Key;
use anyhow::Result;
use application::ticker::Ticker;
use application::App;
use core::Engine;
use log::{debug, info};

struct TestGame {
}

impl TestGame {
    fn new() -> Self {
        TestGame {
        }
    }
}

impl App for TestGame {
    fn init(&mut self, ticker: &mut Ticker, app: &mut Engine) -> Result<()> {
        info!("Init TestGame");
        let window = &mut app.window;
        ticker.set_desired_tick_rate(100.0);
        window.set_visible(true);
        Ok(())
    }

    fn tick(&mut self, _ticker: &mut Ticker, app: &mut Engine) -> Result<()> {
        let window = &mut app.window;

        if window.input().key_pressed(Key::Escape) {
            window.set_mouse_grabbed(!window.is_mouse_grabbed())
        }

        if window.input().key_down(Key::W) {
            debug!("Move");
        }

        // if window.input().mouse_dragged(MouseBtn::Left) {
        //     debug!("Mouse dragged {}", window.input().get_mouse_drag_pixel_distance(MouseBtn::Left));
        // } else {
        //     let vec = window.input().get_mouse_pixel_motion();
        //     if vec.x != 0.0 && vec.y != 0.0 {
        //         debug!("Mouse motion: {vec}");
        //     }
        // }
        
        Ok(())
    }

    fn is_stopped(&self) -> bool {
        false
    }
}

fn main() {
    Engine::start(TestGame::new());
}
