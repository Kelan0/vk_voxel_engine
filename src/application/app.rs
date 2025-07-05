use anyhow::Result;

use crate::application::Ticker;
use crate::Engine;

pub trait App {
    fn init(&mut self, ticker: &mut Ticker, ctx: &mut Engine) -> Result<()>;

    fn tick(&mut self, ticker: &mut Ticker, ctx: &mut Engine) -> Result<()>;

    fn is_stopped(&self) -> bool;
}