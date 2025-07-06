use crate::application::{App, Tickable, Ticker, Window};
use crate::core::Renderer;
use anyhow::Result;
use log::{error, info};
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::format::ClearValue;
use crate::core::renderer::{BeginFrameResult, SecondaryCommandBuffer};

pub struct Engine {
    pub window: Window,
    pub renderer: Renderer,
    user_app: Option<Box<dyn App>>,
}

impl Engine {
    // fn new<T>(user_app: T, update_loop: &'static mut Ticker) -> Result<Self, String>
    // where T: App + 'static {
    fn new<T>(user_app: T) -> Result<Self>
    where T: App + 'static {
        info!("Starting engine");

        let window = Window::new("Test Game", 640, 480)
            .inspect_err(|_| error!("Failed to create application window"))?;

        let renderer = Renderer::new(window.sdl_window_handle())
            .inspect_err(|_| error!("Failed to initialize renderer"))?;

        // let user_app = Some(user_app);
        let user_app = Box::new(user_app);
        let user_app = Some(user_app as Box<dyn App>);

        let app = Engine {
            window,
            renderer,
            user_app,
        };

        Ok(app)
    }

    fn start_internal<T>(user_app: T) -> Result<()>
    where T: App + 'static {
        set_default_env_var("RUST_LOG", "info");
        pretty_env_logger::init();

        let update_loop = Ticker::new(60.0, true);
        let update_loop = Box::leak(Box::new(update_loop)); // update_loop lives forever

        let app = Engine::new(user_app)?;
        let app = Box::leak(Box::new(app)); // app lives forever

        update_loop.add_tickable(app);
        update_loop.start_blocking();

        update_loop.take_result()
    }

    pub fn start<T>(user_app: T)
    where T: App + 'static {
        if let Err(e) = Self::start_internal(user_app) {
            error!("An error occurred during engine execution");
            error!("Root cause: {}", e.root_cause());
            
            let chain = e.chain();
            for e in chain {
                error!("{e}");
            }
            
            // let mut src = e.source();
            // while let Some(e) = src {
            //     error!("{e}");
            //     src = e.source();
            // }
        }
    }

    pub fn user_app(&self) -> &dyn App {
        self.user_app.as_ref().unwrap().as_ref()
    }
}

impl Tickable for Engine {
    fn init(&mut self, ticker: &mut Ticker) -> Result<()> {

        self.renderer.init()?;

        {
            let mut user_app = std::mem::take(&mut self.user_app);
            let result = user_app.as_mut().unwrap().init(ticker, self);
            self.user_app = user_app;
            result?;
        }

        Ok(())
    }

    fn tick(&mut self, ticker: &mut Ticker) -> Result<()> {

        self.window.update();
        if self.window.did_quit() {
            ticker.stop();
        }
        if self.window.did_resize() {
            let size = self.window.get_window_size_in_pixels();
            self.renderer.set_resolution(size.x as u32, size.y as u32)?;
        }

        let clear_values = vec![Some(ClearValue::Float([1.0, 1.0, 0.0, 1.0]))];
        // let clear_values = vec![None];

        if ticker.time_since_last_dbg() >= 1.0 {
            self.renderer.debug_print_ref_counts();
        }
        
        match self.renderer.begin_frame() {
            BeginFrameResult::Begin(mut cmd_buf) => {
                let framebuffer = self.renderer.get_current_framebuffer();
                let render_pass = self.renderer.get_render_pass();


                cmd_buf.begin_render_pass(RenderPassBeginInfo{
                    clear_values,
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                }, SubpassBeginInfo{
                    contents: SubpassContents::Inline,
                    ..Default::default()
                })?;


                cmd_buf.end_render_pass(SubpassEndInfo::default())?;

                self.renderer.present_frame(cmd_buf)?;
            }
            BeginFrameResult::Skip => {}
            BeginFrameResult::Err(err) => return Err(err)
        }
        

        let mut user_app = std::mem::take(&mut self.user_app);
        user_app.as_mut().unwrap().tick(ticker, self)?;
        self.user_app = user_app;
        
        Ok(())
    }

    fn is_stopped(&self) -> bool {
        self.user_app().is_stopped()
    }
}



fn set_default_env_var(key: &str, value: &str) {
    if std::env::var(key).is_err() {
        unsafe {
            std::env::set_var(key, value);
        }
    }
}