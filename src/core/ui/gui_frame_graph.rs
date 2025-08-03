use std::cell::{RefCell, UnsafeCell};
use std::cmp::Ordering;
use std::time::Instant;
use egui::{PopupAnchor, Tooltip};
use egui_plot::PlotItem;
use foldhash::{HashMap, HashMapExt};
use rand::random;
use crate::application::Ticker;
use crate::core::Engine;

pub struct FrameProfiler {
    data: UnsafeCell<ProfilerData>,
    show_profile_stack: bool,
    normalized_stack: bool,
    profile_colours: HashMap<String, egui::Color32>,
    profile_proportions: HashMap<String, f64>,
}

struct ProfilerData {
    all_frame_data: Vec<FrameSlice>,
    frame_indices: Vec<usize>,
    current_frame_stack: Vec<usize>,
}

struct FrameSlice {
    label: String,
    time_start: Instant,
    time_end: Instant,
    level: u16,
    child_offset: u32,
}


pub struct ScopedProfile<'a> {
    // frame_profiler: &'a FrameProfiler,
    frame_profiler: *const FrameProfiler,
    name: &'a str
}

impl<'a> ScopedProfile<'a> {
    pub fn begin(frame_profiler: &FrameProfiler, name: &'a str) -> Self {
        frame_profiler.push_profile(name);

        ScopedProfile {
            frame_profiler: frame_profiler as *const _,
            name
        }
    }
}

impl<'a> Drop for ScopedProfile<'a> {
    fn drop(&mut self) {
        unsafe {
            (*self.frame_profiler).pop_profile(self.name);
        }
    }
}

#[macro_export]
macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        name.strip_suffix("::f").unwrap()

        // // Find and cut the rest of the path
        // match &name[..name.len() - 3].rfind(':') {
        //     Some(pos) => &name[pos + 1..name.len() - 3],
        //     None => &name[..name.len() - 3],
        // }
    }}
}

#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _scoped_profile_guard = $crate::core::ui::ScopedProfile::begin($profiler, $name);
    };
}
#[macro_export]
macro_rules! profile_scope_fn {
    ($profiler:expr) => {
        // let _fn_name = function_name!();
        let _scoped_profile_guard = $crate::core::ui::ScopedProfile::begin($profiler, function_name!());
    };
}

impl FrameProfiler {
    pub fn new() -> Self {
        FrameProfiler {
            data: UnsafeCell::new(ProfilerData {
                all_frame_data: vec![],
                frame_indices: vec![],
                current_frame_stack: vec![],
            }),
            show_profile_stack: false,
            normalized_stack: false,
            profile_colours: HashMap::new(),
            profile_proportions: HashMap::new(),
        }
    }

    pub fn begin_frame(&self) {
        let data = unsafe { &mut *self.data.get() };

        assert_eq!(data.current_frame_stack.len(), 0, "begin_frame() - Profile stack is not empty");

        let index = data.all_frame_data.len();
        data.current_frame_stack.push(index);

        let now = Instant::now();
        data.all_frame_data.push(FrameSlice{
            label: String::from("Frame"),
            time_start: now,
            time_end: now,
            level: 0,
            child_offset: 0,
        });
    }

    pub fn end_frame(&self) {
        assert_eq!(unsafe { &mut *self.data.get() }.current_frame_stack.len(), 1, "end_frame() - Profile stack is not complete");

        let index = self.pop_profile("Frame");

        let data = unsafe { &mut *self.data.get() };
        data.frame_indices.push(index);
    }

    pub fn push_profile(&self, name: &str) {
        let data = unsafe { &mut *self.data.get() };

        let index = data.all_frame_data.len();
        let parent_slice = Self::current_slice(data);

        let now = Instant::now();
        let slice = FrameSlice{
            label: String::from(name),
            time_start: now,
            time_end: now,
            level: parent_slice.level + 1,
            child_offset: 0,
        };

        data.current_frame_stack.push(index);
        data.all_frame_data.push(slice);
    }

    pub fn pop_profile(&self, name: &str) -> usize {
        let data = unsafe { &mut *self.data.get() };

        let slice = Self::current_slice(data);
        slice.time_end = Instant::now();
        debug_assert!(slice.label == name, "pop_profile() - Mismatched profile, we are popping a different profile to what was started");

        let index = data.current_frame_stack.pop().expect("pop_profile() - Profile stack underflow");
        index
    }

    fn current_slice(data: &mut ProfilerData) -> &mut FrameSlice {
        let index = data.current_frame_stack.last().expect("current_slice() - Profile stack underflow");
        data.all_frame_data.get_mut(*index).unwrap()
    }

    pub fn draw_gui(&mut self, ticker: &mut Ticker, ctx: &egui::Context) {
        let data = unsafe { &mut *self.data.get() };

        let plot_height = 220.0;
        let plot_width = 520.0;

        egui::Window::new("Frame Profiler")
            .anchor(egui::Align2::LEFT_BOTTOM, [10.0, 10.0])
            .default_size([plot_width, plot_height])
            .show(ctx, |ui| {

                ui.set_min_size(egui::Vec2::new(plot_width, plot_height));
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.show_profile_stack, "Show Details");
                    if self.show_profile_stack {
                        ui.checkbox(&mut self.normalized_stack, "Normalized");

                        if ui.button("New Colours").clicked() {
                            self.profile_colours.clear();
                        }
                    }
                });



                let max_frame_count = 300;
                let start_index = data.frame_indices.len() - usize::min(data.frame_indices.len(), max_frame_count);
                let recent_frame_indices = &data.frame_indices[start_index..];

                let mut frame_time_bars = //Vec::with_capacity(recent_frame_indices.len());
                    if self.show_profile_stack {
                        let capacity = data.all_frame_data.len() - data.frame_indices[start_index];
                        Vec::with_capacity(capacity)
                    } else {
                        Vec::with_capacity(recent_frame_indices.len())
                    };

                for (index, &offset) in recent_frame_indices.iter().enumerate() {

                    let x = index as f64;

                    if self.show_profile_stack {
                        let root_slice = &data.all_frame_data[offset];
                        let full_dur_millis = root_slice.time_end.duration_since(root_slice.time_start).as_secs_f64() * 1000.0;

                        let mut curr_offset = offset;
                        loop {
                            let curr_slice = &data.all_frame_data[curr_offset];

                            let t0 = curr_slice.time_start.duration_since(root_slice.time_start).as_secs_f64() * 1000.0;
                            let t1 = curr_slice.time_end.duration_since(root_slice.time_start).as_secs_f64() * 1000.0;
                            let dur_millis = t1 - t0;
                            let dur_norm = dur_millis / full_dur_millis;

                            let (h, t0) = if self.normalized_stack {
                                (dur_norm * 100.0, (t0 / full_dur_millis) * 100.0)
                            } else {
                                (dur_millis, t0)
                            };

                            let profile_proportion = self.profile_proportions.entry(curr_slice.label.clone()).or_insert_with(|| dur_norm);

                            const DELTA: f64 = 0.01;
                            *profile_proportion = (dur_norm * DELTA) + (*profile_proportion * (1.0 - DELTA));

                            let bar_colour = self.profile_colours.entry(curr_slice.label.clone()).or_insert_with(|| {
                                let r = random::<u8>();
                                let g = random::<u8>();
                                let b = random::<u8>();
                                let colour = egui::Color32::from_rgb(r, g, b);
                                colour
                            }).clone();

                            let bar = egui_plot::Bar::new(x, h)
                                .name(curr_slice.label.clone())
                                .fill(bar_colour)
                                .base_offset(t0)
                                .stroke(egui::Stroke::new(0.0, bar_colour));
                            frame_time_bars.push(bar);

                            curr_offset += 1;
                            if data.all_frame_data[curr_offset].level == 0 {
                                // We reached the end of the current frame stack (next frame)
                                break;
                            }
                        }

                    } else {
                        let slice = &data.all_frame_data[offset];
                        let dur_millis = slice.time_end.duration_since(slice.time_start).as_secs_f64() * 1000.0;

                        let h = dur_millis;

                        let bar_colour = if dur_millis < (1000.0 / 60.0) {
                            egui::Color32::DARK_GREEN // Above 60 fps
                        } else if dur_millis < (1000.0 / 30.0) {
                            egui::Color32::GREEN // Above 30 fps
                        } else if dur_millis < (1000.0 / 20.0) {
                            egui::Color32::YELLOW // Above 20 fps
                        } else if dur_millis < (1000.0 / 15.0){
                            egui::Color32::ORANGE // Above 15 fps
                        } else {
                            egui::Color32::RED // Below 15 fps
                        };

                        let bar = egui_plot::Bar::new(x, h)
                            .name(format!("{dur_millis} msec"))
                            .fill(bar_colour)
                            .stroke(egui::Stroke::new(0.0, bar_colour));
                        frame_time_bars.push(bar);
                    }
                }

                if self.show_profile_stack {
                    frame_time_bars.sort_by(|a, b| {
                        match a.name.cmp(&b.name) {
                            Ordering::Less => Ordering::Less,
                            Ordering::Greater => Ordering::Greater,
                            Ordering::Equal => a.argument.total_cmp(&b.argument)
                        }
                    });
                }

                let y_axis_label = if self.show_profile_stack && self.normalized_stack {
                    "Frame Time (Percent)"
                } else {
                    "Frame Time (msec)"
                };

                let mut plot = egui_plot::Plot::new("frame_time_graph")
                    // .x_axis_label("Frame")
                    .y_axis_label(y_axis_label)
                    .allow_scroll([true, false])
                    .allow_drag([false, false])
                    .allow_zoom([false, false])
                    .allow_boxed_zoom(false)
                    .auto_bounds([true, true])
                    .show_background(false);

                if self.show_profile_stack {
                    plot = plot.legend(egui_plot::Legend::default());
                }

                plot.show(ui, |plot_ui| {
                    let mut test = vec![];
                    if self.show_profile_stack {
                        let mut start_index = 0;
                        for (index, bar) in frame_time_bars.iter().enumerate() {
                            let is_end = index == (frame_time_bars.len() - 1);
                            let end_index = index + (is_end as usize);

                            if bar.name != frame_time_bars[start_index].name || is_end {
                                let proportion = self.profile_proportions.get(&bar.name).map_or(-1.0, |e| *e * 100.0);
                                let name = format!("{} ({:.2} %)", frame_time_bars[start_index].name, proportion);
                                let v = frame_time_bars[start_index..end_index].to_vec();
                                test.push(v.clone());
                                let bar_chart = egui_plot::BarChart::new(name, v)
                                    .color(frame_time_bars[start_index].fill);
                                plot_ui.bar_chart(bar_chart);
                                start_index = end_index;
                            }
                        }

                        test.len();
                    } else {

                        let bar_chart = egui_plot::BarChart::new("Frame Times", frame_time_bars);
                        plot_ui.bar_chart(bar_chart);
                    }

                    // for bar_chart in frame_time_bar_charts {
                    //     plot_ui.bar_chart(bar_chart);
                    // }
                });

                // for (name, proportion) in self.profile_proportions.iter() {
                //     ui.label(format!("{name}: {:.2}", proportion * 100.0));
                // }

                // bar_chart.find_closest(ctx.pointer_hover_pos().unwrap());
                // let i = r.hovered_plot_item;
            });

    }
}