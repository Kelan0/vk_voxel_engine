use anyhow::{Result};
use log::{debug, error, warn};
use std::thread;
use std::time::{Duration, Instant};

const MIN_SLEEP_DURATION: f64 = 0.001; // 1ms - Sleep if there is at least this amount of time remaining for the current tick.
const DBG_LOG_MISSED_TIME: f64 = 5.0; // 5sec - Log the lost time after accumulating this much difference between simulation time and actual time

pub struct Ticker<'a> {
    running: bool,
    start_time: Instant,
    last_time: Instant,
    last_tick: Instant,
    last_dbg: Instant,
    auto_stop: bool, // Automatically kill the ticker if the tick_list is empty 
    desired_tick_rate: f64,
    measured_tick_rate: f64,
    measured_idle_time: f64,
    delta_time: f64,
    simulation_time: f64,
    real_time: f64,
    idle_time: f64,
    spin_time: f64,
    tick_rate_count: u64,
    last_missed_time: f64,
    accumulated_missed_time: f64,
    last_idle_time: f64,
    partial_tick: f64,
    tick_list: Vec<&'a mut dyn Tickable>,
    result: Option<Result<()>>
}

pub trait Tickable {
    fn init(&mut self, ticker: &mut Ticker) -> Result<()>;

    fn tick(&mut self, ticker: &mut Ticker) -> Result<()>;
    
    fn is_stopped(&self) -> bool;
}

impl<'a> Ticker<'a> {
    pub fn new(desired_tick_rate: f64, auto_stop: bool) -> Self {
        let now = Instant::now();
        
        Ticker {
            running: false,
            start_time: now,
            last_time: now,
            last_tick: now,
            last_dbg: now,
            desired_tick_rate,
            auto_stop,
            measured_tick_rate: 0.0,
            measured_idle_time: 0.0,
            delta_time: 0.0,
            simulation_time: 0.0,
            real_time: 0.0,
            idle_time: 0.0,
            spin_time: 0.0,
            tick_rate_count: 0,
            last_missed_time: 0.0,
            accumulated_missed_time: 0.0,
            last_idle_time: 0.0,
            partial_tick: 0.0,
            tick_list: Vec::new(),
            result: None
        }
    }

    // pub fn add_tickable(&mut self, tickable: Rc<RefCell<dyn Tickable>>) {
    pub fn add_tickable(&mut self, tickable: &'a mut dyn Tickable) {
        // let t = tickable.clone() as Rc<RefCell<dyn Tickable>>;
        self.tick_list.push(tickable);
    }

    pub fn init(&mut self) -> Result<()> {

        self.start_time = Instant::now();
        self.last_time = self.start_time;
        self.last_tick = self.start_time;
        self.last_dbg = self.start_time;

        self.tick_rate_count = 0;

        self.simulation_time = 0.0;
        self.real_time = 0.0;
        self.idle_time = 0.0;
        self.spin_time = 0.0;

        self.measured_tick_rate = 0.0;
        self.measured_idle_time = 0.0;

        self.last_missed_time = 0.0;
        self.accumulated_missed_time = 0.0;

        self.last_idle_time = 0.0;

        self.partial_tick = 0.0;
        self.running = true;

        
        let mut tick_list = std::mem::take(&mut self.tick_list);
        
        for obj in &mut tick_list {
            // let mut obj = obj.borrow_mut();
            // if let Some(obj) = obj.upgrade() {
            //     obj.borrow_mut().init(self)?;
            // }
            obj.init(self)?
        }
        
        self.tick_list = tick_list;
        Ok(())
    }

    fn tick(&mut self) -> Result<()> {
        let mut tick_list = std::mem::take(&mut self.tick_list);
        
        tick_list.retain(|tickable| {
            !tickable.is_stopped()
        });

        for obj in &mut tick_list {
            // let mut obj = obj.borrow_mut();
            // if let Some(obj) = obj.upgrade() {
            //     obj.borrow_mut().tick(self);
            // } else {
            //     // TODO: remove from list
            // }
            obj.tick(self)?;
        }
        
        self.tick_list = tick_list;
        Ok(())
    }
    
    pub fn update(&mut self) -> bool {

        if !self.is_running() {
            return false;
        }
        
        if self.auto_stop && self.tick_list.is_empty() {
            return false;
        }
        
        let tick_start_time = Instant::now();
        let elapsed_time = tick_start_time.duration_since(self.last_time).as_secs_f64();
        self.last_time = tick_start_time;

        let desired_tick_rate = if self.desired_tick_rate < 1e-6 { 1000.0 } else { self.desired_tick_rate };
        let expected_tick_duration: f64 = 1.0 / desired_tick_rate;

        self.partial_tick += elapsed_time / expected_tick_duration;

        if self.partial_tick >= 1.0 {
            self.delta_time = tick_start_time.duration_since(self.last_tick).as_secs_f64();
            self.last_tick = tick_start_time;
            self.partial_tick -= 1.0;

            self.result = Some(self.tick());
            if self.has_error() {
                self.stop();
            }
            
            self.tick_rate_count += 1;
            self.simulation_time += expected_tick_duration;

            let time_since_dbg = tick_start_time.duration_since(self.last_dbg).as_secs_f64();
            if time_since_dbg >= 1.0 {
                self.real_time = tick_start_time.duration_since(self.start_time).as_secs_f64();
                let missed_time = self.real_time - self.simulation_time;
                self.accumulated_missed_time += missed_time - self.last_missed_time;
                self.last_missed_time = missed_time;

                self.measured_tick_rate = (self.tick_rate_count as f64) / time_since_dbg;

                self.measured_idle_time = (self.idle_time - self.last_idle_time) / time_since_dbg;
                self.last_idle_time = self.idle_time;

                debug!("{:.5} UPDATES PER SECOND - {:.5} sec lost - idle for {:.2} sec", self.measured_tick_rate, missed_time, self.measured_idle_time);
                self.last_dbg = tick_start_time;
                self.tick_rate_count = 0;


                if self.accumulated_missed_time >= DBG_LOG_MISSED_TIME {
                    warn!("Can't keep up! Simulation time is {:.2} sec, expected {:.2} sec - we are {:.2} seconds behind!", self.simulation_time, self.real_time, missed_time);
                    self.accumulated_missed_time = 0.0;
                    // TODO: should we skip the list time here?
                }
            }

            let tick_end_time = Instant::now();
            let current_tick_duration = tick_end_time.duration_since(tick_start_time).as_secs_f64();
            let remaining_tick_duration = expected_tick_duration - current_tick_duration;

            if remaining_tick_duration > 0.0 {
                if remaining_tick_duration >= MIN_SLEEP_DURATION {
                    let sleep_micros = (0.5 * remaining_tick_duration * 1_000_000.0) as u64;
                    thread::sleep(Duration::from_micros(sleep_micros));
                } else {
                    // Time this thread is spinning but not sleeping (using CPU cycles to do nothing)
                    self.spin_time += remaining_tick_duration;
                }

                self.idle_time += remaining_tick_duration;
            }
        }
        
        true
    }

    pub fn start_blocking(&mut self) -> bool {
        if let Err(err) = self.init() {
            error!("Failed to start application\nError during initialization: {err}");
            return false;
        }

        while self.update() {}
        
        true
    }

    // pub fn start_thread(&self) -> bool {
    //     let error = false;
    // 
    //     // thread::spawn(move || {
    //     //     self.start_blocking();
    //     // });
    //     //
    //     error
    // }

    pub fn stop(&mut self) {
        self.running = false;
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn get_desired_tick_rate(&self) -> f64 {
        self.desired_tick_rate
    }

    pub fn set_desired_tick_rate(&mut self, desired_tick_rate: f64) {
        self.desired_tick_rate = desired_tick_rate;
    }

    pub fn get_measured_tick_rate(&self) -> f64 {
        self.measured_tick_rate
    }

    pub fn get_measured_idle_time(&self) -> f64 {
        self.measured_idle_time
    }

    pub fn get_delta_time(&self) -> f64 {
        self.delta_time
    }

    pub fn get_simulation_time(&self) -> f64 {
        self.simulation_time
    }

    pub fn get_real_time(&self) -> f64 {
        self.simulation_time
    }

    pub fn get_idle_time(&self) -> f64 {
        self.idle_time
    }
    
    pub fn has_result(&self) -> bool {
        self.result.is_some()
    }
    
    pub fn has_error(&self) -> bool {
        self.has_result() && self.get_result().unwrap().is_err()
    }
    
    pub fn get_result(&self) -> Option<&Result<()>> {
        self.result.as_ref()
    }
    
    pub fn take_result(&mut self) -> Result<()> {
        if !self.has_result() {
            return Ok(());
        }
        
        self.result.take().unwrap()
    }
}