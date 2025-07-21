pub mod util {
    use ash::vk::DeviceSize;
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
}