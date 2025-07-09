use std::backtrace::Backtrace;
use crate::application::{App, Tickable, Ticker, Window};
use crate::core::renderer::{BeginFrameResult, PrimaryCommandBuffer};
use crate::core::GraphicsManager;
use anyhow::Result;
use log::{error, info};
use shrev::ReaderId;
use vulkano::command_buffer::{RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::format::ClearValue;
use crate::application::window::WindowResizedEvent;

pub struct Engine {
    // pub rt: Runtime,
    pub window: Window,
    pub graphics: GraphicsManager,
    user_app: Option<Box<dyn App>>,
    event_window_resized: Option<ReaderId<WindowResizedEvent>>
}

pub struct RenderContext<'a> {
    pub engine: &'a mut Engine,
    pub cmd_buf: &'a mut PrimaryCommandBuffer,
}

impl Engine {

    fn new<T>(user_app: T) -> Result<Self>
    where T: App + 'static {
        info!("Starting engine");

        // info!("Initializing Tokio runtime");
        // let rt = Runtime::new()?;

        let window = Window::new("Test Game", 640, 480)
            .inspect_err(|_| error!("Failed to create application window"))?;

        let graphics = GraphicsManager::new(window.sdl_window_handle())
            .inspect_err(|_| error!("Failed to initialize renderer"))?;

        // let user_app = Some(user_app);
        let user_app = Box::new(user_app);
        let user_app = Some(user_app as Box<dyn App>);

        let app = Engine {
            // rt,
            window,
            graphics,
            user_app,
            event_window_resized: None
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

        // let a = app.user_app.as_ref().unwrap();
        // a.register_events(app)?;

        update_loop.add_tickable(app);
        update_loop.start_blocking();

        update_loop.take_result()
    }

    pub fn start<T>(user_app: T)
    where T: App + 'static {
        if let Err(e) = Self::start_internal(user_app) {
            error!("An error occurred during engine execution");
            let backtrace = e.backtrace();
            error!("Root cause: {}", e.root_cause());

            let chain = e.chain();
            for e in chain {
                error!("{e}");
            }
            
            error!("Backtrace:\n{}", backtrace);

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

    fn register_events(&mut self) {
        self.event_window_resized = Some(self.window.event_bus().register::<WindowResizedEvent>());
    }
    
    fn pre_render(&mut self, ticker: &mut Ticker, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {

        if ticker.time_since_last_dbg() >= 1.0 {
            self.graphics.debug_print_ref_counts();
        }

        let mut user_app = std::mem::take(&mut self.user_app);
        user_app.as_mut().unwrap().pre_render(ticker, self, cmd_buf)?;
        self.user_app = user_app;

        Ok(())
    }

    fn render(&mut self, ticker: &mut Ticker, cmd_buf: &mut PrimaryCommandBuffer) -> Result<()> {
        let clear_values = vec![Some(ClearValue::Float([0.05, 0.05, 0.2, 1.0]))];

        let framebuffer = self.graphics.get_current_framebuffer();

        cmd_buf.begin_render_pass(RenderPassBeginInfo{
            clear_values,
            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
        }, SubpassBeginInfo{
            contents: SubpassContents::Inline,
            ..Default::default()
        })?;

        let mut user_app = std::mem::take(&mut self.user_app);
        user_app.as_mut().unwrap().render(ticker, self, cmd_buf)?;
        self.user_app = user_app;

        cmd_buf.end_render_pass(SubpassEndInfo::default())?;

        Ok(())
    }

}

impl Tickable for Engine {
    fn init(&mut self, ticker: &mut Ticker) -> Result<()> {
        
        self.register_events();

        {
            let mut user_app = std::mem::take(&mut self.user_app);
            let result = user_app.as_mut().unwrap().register_events(self);
            self.user_app = user_app;
            result?;
        }



        self.graphics.init()?;
        
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
        
        if let Some(event) = self.window.event_bus().read_one_opt(&mut self.event_window_resized) {
            self.graphics.set_resolution(event.width, event.height)?;
        }

        match self.graphics.begin_frame() {
            BeginFrameResult::Begin(mut cmd_buf) => {

                self.pre_render(ticker, &mut cmd_buf)?;
                self.render(ticker, &mut cmd_buf)?;

                self.graphics.present_frame(cmd_buf)?;
            }
            BeginFrameResult::Skip => {}
            BeginFrameResult::Err(err) => return Err(err)
        }

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