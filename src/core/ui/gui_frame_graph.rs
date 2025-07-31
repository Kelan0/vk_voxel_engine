use std::time::Instant;
use crate::application::Ticker;
use crate::core::Engine;

pub struct FrameProfiler {
    all_frame_data: Vec<FrameSlice>,
    frame_indices: Vec<usize>,
    current_frame_stack: Vec<usize>,
}

struct FrameSlice {
    label: String,
    time_start: Instant,
    time_end: Instant,
    level: u32,
}

impl FrameProfiler {
    pub fn new() -> Self {
        FrameProfiler {
            all_frame_data: vec![],
            frame_indices: vec![],
            current_frame_stack: vec![],
        }
    }

    pub fn begin_frame(&mut self) {
        assert_eq!(self.current_frame_stack.len(), 0, "begin_frame() - Profile stack is not empty");

        let index = self.all_frame_data.len();
        self.current_frame_stack.push(index);

        let now = Instant::now();
        self.all_frame_data.push(FrameSlice{
            label: String::from("Frame"),
            time_start: now,
            time_end: now,
            level: 0,
        });
    }

    pub fn end_frame(&mut self) {
        assert_eq!(self.current_frame_stack.len(), 1, "end_frame() - Profile stack is not complete");

        let index = self.pop_profile("Frame");
        self.frame_indices.push(index);
    }

    pub fn push_profile(&mut self, name: &str) {
        let index = self.all_frame_data.len();
        let parent_slice = self.current_slice();

        let now = Instant::now();
        let slice = FrameSlice{
            label: String::from(name),
            time_start: now,
            time_end: now,
            level: parent_slice.level + 1,
        };

        self.current_frame_stack.push(index);
        self.all_frame_data.push(slice);
    }

    pub fn pop_profile(&mut self, name: &str) -> usize {
        let slice = self.current_slice();
        slice.time_end = Instant::now();
        debug_assert!(slice.label == name, "pop_profile() - Mismatched profile, we are popping a different profile to what was started");

        let index = self.current_frame_stack.pop().expect("pop_profile() - Profile stack underflow");
        index
    }

    fn current_slice(&mut self) -> &mut FrameSlice {
        let index = self.current_frame_stack.last().expect("current_slice() - Profile stack underflow");
        self.all_frame_data.get_mut(*index).unwrap()
    }

    pub fn draw_gui(&self, ticker: &mut Ticker, ctx: &egui::Context) {

        let plot_height = 220.0;
        let plot_width = 520.0;

        egui::Window::new("Frame Profiler")
            .anchor(egui::Align2::LEFT_BOTTOM, [10.0, 10.0])
            .default_size([plot_width, plot_height])
            .show(ctx, |ui| {
                let mut frame_time_bars = vec![];

                ui.set_min_size(egui::Vec2::new(plot_width, plot_height));

                let max_frame_count = 300;
                let start_index = self.frame_indices.len() - usize::min(self.frame_indices.len(), max_frame_count);
                let recent_frame_indices = &self.frame_indices[start_index..];

                for (index, &offset) in recent_frame_indices.iter().enumerate() {
                    let slice = &self.all_frame_data[offset];
                    let dur_millis = slice.time_end.duration_since(slice.time_start).as_secs_f64() * 1000.0;

                    let x = index as f64;
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

                let plot = egui_plot::Plot::new("frame_time_graph")
                    // .x_axis_label("Frame")
                    .y_axis_label("Frame Time (msec)")
                    .allow_scroll([true, false])
                    .allow_drag([false, false])
                    .allow_zoom([false, false])
                    .allow_boxed_zoom(false)
                    .auto_bounds([true, true]);

                plot.show(ui, |plot_ui| {
                    let bar_chart = egui_plot::BarChart::new("Frame Times", frame_time_bars);
                    plot_ui.bar_chart(bar_chart);
                });
            });

    }
}