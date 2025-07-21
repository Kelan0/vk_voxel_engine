use crate::core::{CommandBuffer, CommandBufferImpl, Engine, GraphicsManager};
use anyhow::Result;
use png::{BitDepth, ColorType, OutputInfo};
use smallvec::smallvec;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{BufferImageCopy, CopyBufferToImageInfo};
use vulkano::device::{Device, DeviceOwned, DeviceOwnedVulkanObject};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, LOD_CLAMP_NONE};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageTiling, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator};
use vulkano::DeviceSize;

#[derive(Clone, Debug, Eq)]
pub struct Texture {
    resource_id: u64,
    image_view: Arc<ImageView>,
    sampler: Arc<Sampler>
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

    pub fn new_from_image(image: Arc<Image>, sampler: Arc<Sampler>) -> Result<Self> {
        let image_view = Texture::create_image_view_from_image(image)?;
        Ok(Self::new(image_view, sampler))
    }

    pub fn resource_id(&self) -> u64 {
        self.resource_id
    }

    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    pub fn sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }

    pub fn image(&self) -> &Arc<Image> {
        self.image_view.image()
    }

    pub fn extent(&self) -> [u32; 3] {
        self.image_view.image().extent()
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

    pub fn load_image_data_from_buffer<T>(cmd_buf: &mut CommandBuffer, buffer: &Subbuffer<[T]>, image: Arc<Image>) -> Result<()>
    where T: BufferContents + Sized {

        let copy_info = CopyBufferToImageInfo{
            ..CopyBufferToImageInfo::buffer_image(buffer.clone(), image.clone())
        };

        cmd_buf.copy_buffer_to_image(copy_info)?;

        Ok(())
    }

    pub fn load_image_region_data_from_buffer<T>(cmd_buf: &mut CommandBuffer, buffer: &Subbuffer<[T]>, dst_image: Arc<Image>, dst_offset: [u32; 3], dst_extent: [u32; 3]) -> Result<()>
    where T: BufferContents + Sized {

        let copy_region = BufferImageCopy {
            image_subresource: dst_image.subresource_layers(),
            image_offset: dst_offset,
            image_extent: dst_extent,
            ..Default::default()
        };

        let copy_info = CopyBufferToImageInfo{
            regions: smallvec![copy_region],
            ..CopyBufferToImageInfo::buffer_image(buffer.clone(), dst_image.clone())
        };

        cmd_buf.copy_buffer_to_image(copy_info)?;

        Ok(())
    }

    pub fn load_image_from_data_staged<T>(cmd_buf: &mut CommandBuffer, staging_buffer: &Subbuffer<[T]>, data: &[T], image: Arc<Image>) -> Result<()>
    where T: BufferContents + Sized + Clone {

        GraphicsManager::upload_buffer_data_sized(staging_buffer, data)?;
        Self::load_image_data_from_buffer(cmd_buf, staging_buffer, image)?;

        Ok(())
    }

    pub fn load_image_region_from_data_staged<T>(cmd_buf: &mut CommandBuffer, staging_buffer: &Subbuffer<[T]>, data: &[T], dst_image: Arc<Image>, dst_offset: [u32; 3], dst_extent: [u32; 3]) -> Result<()>
    where T: BufferContents + Sized + Clone {

        GraphicsManager::upload_buffer_data_sized(staging_buffer, data)?;
        Self::load_image_region_data_from_buffer(cmd_buf, staging_buffer, dst_image, dst_offset, dst_extent)?;

        Ok(())
    }

    pub fn load_image_from_file_staged(cmd_buf: &mut CommandBuffer, allocator: Arc<dyn MemoryAllocator>, staging_buffer: &Subbuffer<[u8]>, file_path: &str, usage: ImageUsage) -> Result<Arc<Image>> {

        let mut buf = vec![];
        let info = Self::load_image_file(file_path, 0, &mut buf)?;
        let bytes = &buf[..info.buffer_size()];
        let format = Self::get_image_format(&info);

        let image = Self::create_image_2d(allocator, info.width, info.height, format, usage | ImageUsage::TRANSFER_DST)?;

        Self::load_image_from_data_staged(cmd_buf, staging_buffer, bytes, image.clone())?;

        Ok(image)
    }

    pub fn load_image_region_from_file_staged(cmd_buf: &mut CommandBuffer, staging_buffer: &Subbuffer<[u8]>, file_path: &str, dst_image: Arc<Image>, dst_offset: [u32; 3], dst_extent: [u32; 3]) -> Result<()> {

        let mut buf = vec![];
        let info = Self::load_image_file(file_path, 0, &mut buf)?;
        let bytes = &buf[..info.buffer_size()];
        let format = Self::get_image_format(&info);
        assert_eq!(format, dst_image.format());

        Self::load_image_region_from_data_staged(cmd_buf, staging_buffer, bytes, dst_image, dst_offset, dst_extent)
    }

    pub fn load_image_file(file_path: &str, offset: usize, buf: &mut Vec<u8>) -> Result<OutputInfo> {

        let file = File::open(file_path)?;
        let decoder = png::Decoder::new(file);
        // decoder.set_transformations(Transformations::IDENTITY)
        let mut reader = decoder.read_info()?;

        buf.resize(offset + reader.output_buffer_size(), 0);
        let info = reader.next_frame(buf)?;

        Ok(info)
    }

    pub fn get_image_format(info: &OutputInfo) -> Format {
        match info {
            OutputInfo { color_type, bit_depth, .. } if *color_type == ColorType::Grayscale && *bit_depth == BitDepth::Eight => Format::R8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if *color_type == ColorType::Grayscale && *bit_depth == BitDepth::Sixteen => Format::R16_UNORM,
            OutputInfo { color_type, bit_depth, .. } if *color_type == ColorType::GrayscaleAlpha && *bit_depth == BitDepth::Eight => Format::R8G8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if *color_type == ColorType::Rgb && *bit_depth == BitDepth::Eight => Format::R8G8B8_UNORM,
            OutputInfo { color_type, bit_depth, .. } if *color_type == ColorType::Rgba && *bit_depth == BitDepth::Eight => Format::R8G8B8A8_UNORM,
            _ => Format::UNDEFINED
        }
    }
}

impl PartialEq for Texture {
    fn eq(&self, other: &Self) -> bool {
        self.resource_id == other.resource_id
    }
}

impl PartialOrd for Texture {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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


#[derive(Debug, Clone, PartialEq)]
pub struct TextureAtlas {
    texture: Texture,
    texture_size: u32,
    column_count: u32,
    row_count: u32,
    atlas_width: u32,
    atlas_height: u32,
    region_map: HashMap<String, (u32, u32)>,
    loading_ctx: Option<TextureAtlasLoadingContext>
}

#[derive(Debug, Clone, PartialEq)]
struct TextureAtlasLoadingContext {
    staging_buffer: Subbuffer<[u8]>,
    offset: DeviceSize,
    stride: DeviceSize,
}

impl TextureAtlas {
    pub fn new(allocator: Arc<dyn MemoryAllocator>, texture_size: u32, column_count: u32, row_count: u32, format: Format, usage: ImageUsage) -> Result<Self> {
        let device = allocator.device().clone();

        let atlas_width = texture_size * column_count;
        let atlas_height = texture_size * row_count;
        let image = Texture::create_image_2d(allocator, atlas_width, atlas_height, format, usage)?;

        let sampler = Texture::create_default_sampler(device)?;
        let texture = Texture::new_from_image(image, sampler)?;

        let texture_atlas = TextureAtlas{
            texture,
            texture_size,
            column_count,
            row_count,
            atlas_width,
            atlas_height,
            region_map: HashMap::new(),
            loading_ctx: None
        };

        Ok(texture_atlas)
    }
    
    pub fn create_staging_buffer(&self, allocator: Arc<dyn MemoryAllocator>) -> Result<Subbuffer<[u8]>> {
        let required_len = (self.atlas_width * self.atlas_height * 4) as DeviceSize;
        let buf = GraphicsManager::create_staging_subbuffer(allocator, required_len)?;
        buf.buffer().set_debug_utils_object_name(Some("TextureAtlas-StagingBuffer"))?;
        Ok(buf)
    }

    pub fn begin_loading(&mut self, staging_buffer: Subbuffer<[u8]>) -> Result<()> {
        debug_assert_eq!(self.loading_ctx, None);

        let stride = (self.texture_size * self.texture_size * 4) as DeviceSize;
        
        self.loading_ctx = Some(TextureAtlasLoadingContext{
            staging_buffer,
            offset: 0,
            stride
        });
        Ok(())
    }

    pub fn finish_loading(&mut self) {
        debug_assert_ne!(self.loading_ctx, None);
        self.loading_ctx = None;
    }

    pub fn load_texture_from_data<L>(&mut self, cmd_buf: &mut CommandBuffer, key: &str, column: u32, row: u32, data: &[u8]) -> Result<()> {
        let ctx = self.loading_ctx.as_mut().unwrap();
        let staging_buffer = ctx.staging_buffer.clone().slice(ctx.offset .. ctx.offset+ctx.stride);
        ctx.offset += ctx.stride;
        let image = self.texture.image().clone();
        let dst_offset = [column * self.texture_size, row * self.texture_size, 0];
        let dst_extent = [self.texture_size, self.texture_size, 1];
        Texture::load_image_region_from_data_staged(cmd_buf, &staging_buffer, data, image, dst_offset, dst_extent)?;
        self.region_map.insert(key.to_string(), (column, row));
        Ok(())
    }

    pub fn load_texture_from_file(&mut self, cmd_buf: &mut CommandBuffer, key: &str, column: u32, row: u32, file_path: &str) -> Result<()>{
        let ctx = self.loading_ctx.as_mut().unwrap();
        let staging_buffer = ctx.staging_buffer.clone().slice(ctx.offset .. ctx.offset+ctx.stride);
        ctx.offset += ctx.stride;
        let image = self.texture.image().clone();
        let dst_offset = [column * self.texture_size, row * self.texture_size, 0];
        let dst_extent = [self.texture_size, self.texture_size, 1];
        Texture::load_image_region_from_file_staged(cmd_buf, &staging_buffer, file_path, image, dst_offset, dst_extent)?;
        self.region_map.insert(key.to_string(), (column, row));
        Ok(())
    }
    
    pub fn get_coords_for_cell(&self, column: u32, row: u32) -> [[f32; 2]; 2] {
        let rw = 1.0 / self.atlas_width as f32;
        let rh = 1.0 / self.atlas_height as f32;
        let x0 = (column * self.texture_size) as f32 * rw;
        let y0 = (row * self.texture_size) as f32 * rh;
        let x1 = ((column + 1) * self.texture_size) as f32 * rw;
        let y1 = ((row + 1) * self.texture_size) as f32 * rh;
        [[x0, y0], [x1, y1]]
    }
    
    pub fn find_coords_for_cell(&self, key: &str) -> Option<[[f32; 2]; 2]> {
        self.region_map.get(key).map(|&(column, row)| {
            self.get_coords_for_cell(column, row)
        })
    }

    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    pub fn texture_size(&self) -> u32 {
        self.texture_size
    }

    pub fn column_count(&self) -> u32 {
        self.column_count
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    pub fn atlas_width(&self) -> u32 {
        self.atlas_width
    }

    pub fn atlas_height(&self) -> u32 {
        self.atlas_height
    }
}