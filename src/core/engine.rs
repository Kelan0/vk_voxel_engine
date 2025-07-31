use crate::application::window::WindowResizedEvent;
use crate::application::{App, Tickable, Ticker, Window};
use crate::core::renderer::BeginFrameResult;
use crate::core::scene::Scene;
use crate::core::{debug_mesh, CommandBuffer, CommandBufferImpl, GraphicsManager, SceneRenderer, GUIRenderer, FrameProfiler};
use anyhow::Result;
use log::{error, info};
use shrev::ReaderId;
use vulkano::command_buffer::{
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::format::ClearValue;

// Eww global state in Rust >:(
static mut NEXT_RESOURCE_ID: u64 = 0;


pub struct Context {}
pub struct Engine {
    // pub rt: Runtime,
    pub window: Window,
    pub graphics: GraphicsManager,
    pub scene: Scene,
    pub scene_renderer: SceneRenderer,
    pub ui_handler: GUIRenderer,
    pub frame_profiler: FrameProfiler,
    user_app: Option<Box<dyn App>>,
    event_window_resized: Option<ReaderId<WindowResizedEvent>>,
    next_resource_id: u64,
}

impl Engine {
    fn new<T>(user_app: T) -> Result<Self>
    where
        T: App + 'static,
    {
        info!("Starting engine");

        // info!("Initializing Tokio runtime");
        // let rt = Runtime::new()?;

        let window = Window::new("Test Game", 640, 480)
            .inspect_err(|_| error!("Failed to create application window"))?;

        let graphics = GraphicsManager::new(window.sdl_window_handle())
            .inspect_err(|_| error!("Failed to create GraphicsManager"))?;

        let scene = Scene::new()
            .inspect_err(|_| error!("Failed to create Scene"))?;

        let scene_renderer = SceneRenderer::new(&graphics)
            .inspect_err(|_| error!("Failed to create SceneRenderer"))?;

        let ui_handler = GUIRenderer::new(&graphics)
            .inspect_err(|_| error!("Failed to create UIHandler"))?;

        let frame_profiler = FrameProfiler::new();

        // let user_app = Some(user_app);
        let user_app = Box::new(user_app);
        let user_app = Some(user_app as Box<dyn App>);

        let app = Engine {
            // rt,
            window,
            graphics,
            scene,
            scene_renderer,
            ui_handler,
            frame_profiler,
            user_app,
            event_window_resized: None,
            next_resource_id: 0,
        };

        Ok(app)
    }

    fn start_internal<T>(user_app: T) -> Result<()>
    where
        T: App + 'static,
    {
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


        let result = update_loop.take_result();
            
        app.shutdown();

        result
    }

    pub fn start<T>(user_app: T)
    where
        T: App + 'static,
    {
        if let Err(e) = Self::start_internal(user_app) {
            error!("An error occurred during engine execution");
            let backtrace = e.backtrace();
            error!("Root cause: {}", e.root_cause());

            let chain = e.chain();
            for e in chain {
                error!("{e}");
            }

            error!("Backtrace:\n{backtrace}");

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

    pub fn user_app_mut(&mut self) -> &mut Box<dyn App> {
        self.user_app.as_mut().unwrap()
    }

    fn register_events(&mut self) -> Result<()> {
        self.event_window_resized = Some(self.window.event_bus().register::<WindowResizedEvent>());

        // dirty hack ;)
        let self_ptr: *mut Self = self;

        unsafe { self.user_app_mut().register_events(&mut *self_ptr) }?;

        unsafe { self.scene_renderer.register_events(&mut *self_ptr) }?;

        unsafe { self.ui_handler.register_events(&mut *self_ptr) }?;

        Ok(())
    }

    fn pre_render(
        &mut self,
        ticker: &mut Ticker,
        cmd_buf: &mut CommandBuffer,
    ) -> Result<()> {
        if ticker.time_since_last_dbg() >= ticker.debug_interval() {
            self.graphics.debug_print_ref_counts();
        }

        // dirty hack ;)
        let self_ptr: *mut Self = self;
        

        unsafe { self.user_app_mut().pre_render(ticker, &mut *self_ptr, cmd_buf) }?;

        unsafe { self.scene_renderer.pre_render(ticker, &mut *self_ptr, cmd_buf) }?;

        unsafe { self.scene.pre_render(ticker, &mut *self_ptr) }?;

        Ok(())
    }

    fn render(&mut self, ticker: &mut Ticker, cmd_buf: &mut CommandBuffer) -> Result<()> {
        let clear_values = vec![
            Some(ClearValue::Float([0.05, 0.05, 0.2, 1.0])),
            Some(ClearValue::Depth(1.0))
        ];

        let framebuffer = self.graphics.current_framebuffer();

        cmd_buf.begin_render_pass(
            RenderPassBeginInfo {
                clear_values,
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?;

        // dirty hack ;)
        let self_ptr: *mut Self = self;

        unsafe { self.user_app_mut().render(ticker, &mut *self_ptr, cmd_buf) }?;

        unsafe { self.scene_renderer.render(ticker, &mut *self_ptr, cmd_buf) }?;

        unsafe { self.scene.render(ticker, &mut *self_ptr) }?;

        unsafe { self.ui_handler.render_gui(ticker, &mut *self_ptr, cmd_buf, self.graphics.resolution(), Self::run_gui)?; }

        cmd_buf.end_render_pass(SubpassEndInfo::default())?;

        Ok(())
    }

    fn run_gui(ticker: &mut Ticker, engine: &mut Engine, ctx: &egui::Context) {
        engine.draw_gui(ticker, ctx);
    }

    fn draw_gui(&mut self, ticker: &mut Ticker, ctx: &egui::Context) {
        egui::Window::new("Test Window").show(ctx, |ui| {

            ui.label("Hello World");
            if ui.button("Test").clicked() {
                info!("Clicked a button!");
            }

            ctx.texture_ui(ui)
        });

        self.frame_profiler.draw_gui(ticker, ctx);
    }
    
    fn shutdown(&mut self) {
        self.user_app_mut().shutdown();
    }

    /// Get a unique identifier for any resource that needs to be tracked
    /// Resource IDs start at 1. zero is an invalid ID and can be used as default.
    pub fn next_resource_id() -> u64 {
        // self.next_resource_id += 1;
        // self.next_resource_id

        unsafe {
            NEXT_RESOURCE_ID += 1;
            NEXT_RESOURCE_ID
        }
    }
}

impl Tickable for Engine {
    fn init(&mut self, ticker: &mut Ticker) -> Result<()> {
        self.register_events()?;

        // dirty hack ;)
        let self_ptr: *mut Self = self;

        self.graphics.init()?;

        debug_mesh::init(&self.graphics)?;

        self.ui_handler.init(&self.graphics)?;

        unsafe { self.scene_renderer.init(&mut *self_ptr) }?;

        unsafe { self.user_app_mut().init(ticker, &mut *self_ptr) }?;

        Ok(())
    }

    fn tick(&mut self, ticker: &mut Ticker) -> Result<()> {
        self.window.update(|event, input_handler| {
            self.ui_handler.process_event(event, input_handler);
        });

        if self.window.did_quit() {
            ticker.stop();
        }

        if let Some(event) = self.window.event_bus().read_one_opt_ref(&mut self.event_window_resized) {
            self.graphics.set_resolution(event.width, event.height)?;
        }

        match self.graphics.begin_frame() {
            BeginFrameResult::Begin(mut cmd_buf) => {
                self.frame_profiler.begin_frame();
                self.pre_render(ticker, &mut cmd_buf)?;
                self.render(ticker, &mut cmd_buf)?;

                self.graphics.present_frame(cmd_buf)?;
                self.frame_profiler.end_frame();
            }
            BeginFrameResult::Skip => {}
            BeginFrameResult::Err(err) => return Err(err),
        }

        Ok(())
    }

    fn is_stopped(&self) -> bool {
        self.user_app().is_stopped()
    }
}

pub fn set_default_env_var(key: &str, value: &str) {
    if std::env::var(key).is_err() {
        unsafe {
            std::env::set_var(key, value);
        }
    }
}
