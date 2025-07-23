
#[macro_export]
macro_rules! log_error_and_anyhow {
    ($msg:literal $(, $arg:expr)*) => {{
        let formatted = format!($msg $(, $arg)*);
        error!("{}", formatted);
        anyhow!(formatted)
    }};
}

#[macro_export]
macro_rules! log_error_and_throw {
    ($err:expr, $msg:literal $(, $arg:expr)*) => {{
        error!($msg, $($arg)*);
        $err
    }};
}


use std::ops::{Add, RangeBounds};
use ash::vk::DeviceSize;
use num::{One, Zero};
use vulkano::buffer::Subbuffer;

pub fn get_raw_bytes<T>(data: &T) -> &[u8] {
    let ptr = data as *const T as *const u8;
    let len = size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub fn get_raw_bytes_slice<T>(data: &[T]) -> &[u8] {
    let ptr = data as *const [T] as *const u8;
    let len = size_of_val(data);
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub fn split_buffer_option<T>(buffer: Subbuffer<[T]>, mid: DeviceSize) -> (Subbuffer<[T]>, Option<Subbuffer<[T]>>) {

    if mid >= buffer.len() {
        (buffer, None)
    } else {
        let (a, b) = buffer.split_at(mid);
        (a, Some(b))
    }
}

pub fn chop_buffer_at<T>(buffer: &mut Option<Subbuffer<[T]>>, mid: DeviceSize) -> Option<Subbuffer<[T]>> {
    if let Some(buf) = buffer {
        let (first, remainder) = split_buffer_option(buf.clone(), mid);
        *buffer = remainder;
        return Some(first);
    }
    None
}

pub fn get_range<R, T>(range: R, len: T) -> (T, T)
where R: RangeBounds<T>,
      T: Clone + Copy + One + Zero {

    let start = match range.start_bound() {
        std::ops::Bound::Included(&s) => s,
        std::ops::Bound::Excluded(&s) => s + One::one(),
        std::ops::Bound::Unbounded => Zero::zero(),
    };

    let end = match range.end_bound() {
        std::ops::Bound::Included(&e) => e + One::one(), // inclusive range ends at e+1
        std::ops::Bound::Excluded(&e) => e,
        std::ops::Bound::Unbounded => len,
    };

    (start, end)
}

pub fn f32_to_u16_norm<const L: usize>(coord: [f32; L]) -> [u16; L] {
    const M: f32 = u16::MAX as f32;
    coord.map(|x| (x * M).round() as u16)
}