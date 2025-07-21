use anyhow::Result;

use crate::application::Ticker;
use crate::core::CommandBuffer;
use crate::Engine;

pub trait App {
    fn register_events(&mut self, engine: &mut Engine) -> Result<()>;
    
    fn init(&mut self, ticker: &mut Ticker, engine: &mut Engine) -> Result<()>;

    fn pre_render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()>;
    
    fn render(&mut self, ticker: &mut Ticker, engine: &mut Engine, cmd_buf: &mut CommandBuffer) -> Result<()>;
    
    fn shutdown(&mut self) {}

    fn is_stopped(&self) -> bool;
}