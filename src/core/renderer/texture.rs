use crate::core::{CommandBuffer, Engine, GraphicsManager};
use anyhow::Result;
use png::{BitDepth, ColorType, OutputInfo};
use std::cmp::Ordering;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::CopyBufferToImageInfo;
use vulkano::command_buffer::ResourceInCommand::ImageMemoryBarrier;
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, LOD_CLAMP_NONE};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageTiling, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator};

#[derive(Clone, Debug, Eq)]
pub struct Texture {
    resource_id: u64,
    image_view: Arc<ImageView>,
    sampler: Arc<Sampler>
}

#[repr(C)]
pub enum PixelDataFormat {
    RGBu8(u8, u8, u8),
    RGBAu8(u8, u8, u8, u8),
    RGBf32(f32, f32, f32),
    RGBAf32(f32, f32, f32, f32),
}

#[derive(BufferContents)]
#[repr(C)]
pub struct PixelData {
    r: u8,
    g: u8,
    b: u8,
    a: u8
}

impl Texture {
    pub fn new(image_view: Arc<ImageView>, sampler: Arc<Sampler>) -> Self {
        let resource_id = Engine::next_resource_id();

        Texture{
            resource_id,
            image_view,
            sampler
        }
    }

    pub fn resource_id(&self) -> u64 {
        self.resource_id
    }

    pub fn image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }

    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    pub fn create_image_2d(allocator: Arc<dyn MemoryAllocator>, width: u32, height: u32, format: Format, usage: ImageUsage) -> Result<Arc<Image>> {

        let image_create_info = ImageCreateInfo{
            format,
            image_type: ImageType::Dim2d,
            extent: [width, height, 1],
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage,
            initial_layout: ImageLayout::Undefined,
            ..Default::default()
        };

        let allocation_info = AllocationCreateInfo{
            ..Default::default()
        };
        let image = Image::new(allocator, image_create_info, allocation_info)?;

        Ok(image)
    }

    pub fn create_image_view_2d(allocator: Arc<dyn MemoryAllocator>, width: u32, height: u32, format: Format, usage: ImageUsage) -> Result<Arc<ImageView>> {
        let image = Self::create_image_2d(allocator, width, height, format, usage)?;
        Self::create_image_view_from_image(image)
    }
    
    pub fn create_image_view_from_image(image: Arc<Image>) -> Result<Arc<ImageView>> {

        let image_view_create_info = ImageViewCreateInfo{
            ..ImageViewCreateInfo::from_image(image.as_ref())
        };

        let image_view = ImageView::new(image, image_view_create_info)?;

        Ok(image_view)
    }

    pub fn create_default_sampler(device: Arc<Device>) -> Result<Arc<Sampler>> {
        let sampler_create_info = SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            lod: 0.0..=LOD_CLAMP_NONE,
            ..Default::default()
        };
        let sampler = Sampler::new(device, sampler_create_info)?;
        Ok(sampler)
    }

    pub fn load_image_data_from_buffer<L, T>(cmd_buf: &mut CommandBuffer<L>, buffer: Subbuffer<[T]>, image: Arc<Image>) -> Result<()>
    where T: BufferContents + Sized {
        
        let copy_info = CopyBufferToImageInfo{
            dst_image_layout: ImageLayout::TransferDstOptimal,
            ..CopyBufferToImageInfo::buffer_image(buffer.clone(), image.clone())
        };
        
        
        cmd_buf.copy_buffer_to_image(copy_info)?;

        Ok(())
    }

    pub fn load_image_from_data_staged<L, T>(cmd_buf: &mut CommandBuffer<L>, staging_buffer: Subbuffer<[T]>, data: &[T], image: Arc<Image>) -> Result<()>
    where T: BufferContents + Sized + Clone {

        GraphicsManager::upload_buffer_data_sized(staging_buffer.clone(), data)?;
        Self::load_image_data_from_buffer(cmd_buf, staging_buffer, image)?;

        Ok(())
    }

    pub fn load_image_from_file_staged<L>(cmd_buf: &mut CommandBuffer<L>, allocator: Arc<dyn MemoryAllocator>, staging_buffer: Subbuffer<[u8]>, file_path: &str, usage: ImageUsage) -> Result<Arc<Image>> {
        
        let mut buf = vec![];
        let info = Self::load_image_file(file_path, 0, &mut buf)?;

        let bytes = &buf[..info.buffer_size()];
        
        let format = match info {
            OutputInfo { color_type, bit_depth, .. } if color_type == ColorType::Grayscale && bit_depth == BitDepth::Eight => Format::R8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if color_type == ColorType::Grayscale && bit_depth == BitDepth::Sixteen => Format::R16_UNORM,
            OutputInfo { color_type, bit_depth, .. } if color_type == ColorType::GrayscaleAlpha && bit_depth == BitDepth::Eight => Format::R8G8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if color_type == ColorType::Rgb && bit_depth == BitDepth::Eight => Format::R8G8B8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if color_type == ColorType::Rgba && bit_depth == BitDepth::Eight => Format::R8G8B8A8_UNORM,
            _ => Format::UNDEFINED
        };
        
        let usage = usage | ImageUsage::TRANSFER_DST;
        let image = Self::create_image_2d(allocator, info.width, info.height, format, usage)?;
        
        Self::load_image_from_data_staged(cmd_buf, staging_buffer, bytes, image.clone())?;
        
        Ok(image)
    }
    
    pub fn load_image_file(file_path: &str, offset: usize, buf: &mut Vec<u8>) -> Result<OutputInfo> {

        let file = File::open(file_path)?;
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info()?;

        buf.resize(offset + reader.output_buffer_size(), 0);
        let info = reader.next_frame(buf)?;
        
        Ok(info)
    }
}

impl PartialEq for Texture {
    fn eq(&self, other: &Self) -> bool {
        self.resource_id == other.resource_id
    }
}

impl PartialOrd for Texture {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.resource_id.partial_cmp(&other.resource_id)
    }
}

impl Ord for Texture {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resource_id.cmp(&other.resource_id)
    }
}

impl Hash for Texture {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resource_id.hash(state);
    }
}