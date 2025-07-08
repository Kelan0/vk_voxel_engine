use anyhow::Result;

use crate::application::Ticker;
use crate::core::PrimaryCommandBuffer;
use crate::Engine;

pub trait App {
    fn register_events(&mut self, engine: &mut Engine) -> Result<()>;
    
    fn init(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()>;

    fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()>;
    
    fn render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()>;

    fn is_stopped(&self) -> bool;
}