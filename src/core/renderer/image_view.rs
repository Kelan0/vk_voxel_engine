use std::sync::Arc;
use anyhow::Result;
use log::error;
use vulkano::format::Format;
use vulkano::image::{Image, ImageAspects, ImageSubresourceRange, ImageUsage};
use vulkano::image::sampler::ComponentMapping;
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};

type VkImageView = vulkano::image::view::ImageView;

pub struct ImageViewConfiguration {
    pub view_type: ImageViewType,
    pub format: Format,
    pub component_mapping: ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
    pub usage: ImageUsage,
}

impl ImageViewConfiguration {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set_view_type(mut self, view_type: ImageViewType) -> Self {
        self.view_type = view_type;
        self
    }

    pub fn set_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn set_component_mapping(mut self, component_mapping: ComponentMapping) -> Self {
        self.component_mapping = component_mapping;
        self
    }

    pub fn set_subresource_range(mut self, subresource_range: ImageSubresourceRange) -> Self {
        self.subresource_range = subresource_range;
        self
    }

    pub fn set_usage(mut self, usage: ImageUsage) -> Self {
        self.usage = usage;
        self
    }
}

impl Default for ImageViewConfiguration {
    fn default() -> Self {
        ImageViewConfiguration{
            view_type: ImageViewType::Dim2d,
            format: Format::default(),
            component_mapping: ComponentMapping::default(),
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::empty(),
                array_layers: 0..0,
                mip_levels: 0..0,
            },
            usage: ImageUsage::default(),
        }
    }
}

pub struct ImageView {
    image_view: Arc<VkImageView>,
}

impl ImageView {
    pub fn new(image: Arc<Image>, config: ImageViewConfiguration) -> Result<Self> {
        let create_info = ImageViewCreateInfo{
            view_type: config.view_type,
            format: config.format,
            component_mapping: config.component_mapping,
            subresource_range: config.subresource_range,
            usage: config.usage,
            ..Default::default()
        };

        let image_view = VkImageView::new(image, create_info)
            .inspect_err(|err| error!("Failed to create ImageView: {err}"))?;

        Ok(ImageView{
            image_view
        })
    }

    pub fn get(&self) -> &Arc<VkImageView> {
        &self.image_view
    }

    pub fn image(&self) -> &Arc<Image> {
        self.image_view.image()
    }

    pub fn view_type(&self) -> ImageViewType {
        self.image_view.view_type()
    }

}