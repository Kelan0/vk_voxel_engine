use crate::core::renderer::render_pass::VkRenderPass;
use anyhow::Result;
use std::sync::Arc;
use log::error;
use vulkano::image::view::ImageView;
use vulkano::render_pass::FramebufferCreateInfo;
use vulkano::VulkanObject;

#[derive(Default, Clone, Debug)]
pub struct FramebufferConfiguration {
    pub resolution: [u32; 2],
    pub layers: u32,
    pub attachments: Vec<Arc<ImageView>>
}

impl FramebufferConfiguration {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set_resolution(mut self, resolution: [u32; 2]) -> Self {
        self.resolution = resolution;
        self
    }

    pub fn set_width(mut self, width: u32) -> Self {
        self.resolution[0] = width;
        self
    }

    pub fn set_height(mut self, height: u32) -> Self {
        self.resolution[1] = height;
        self
    }

    pub fn set_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    pub fn set_attachments(mut self, attachments: &[Arc<ImageView>]) -> Self {
        self.attachments = attachments.to_vec();
        self
    }
    
    pub fn add_attachment(mut self, attachment: Arc<ImageView>) -> Self {
        self.attachments.push(attachment);
        self
    }
}

impl From<FramebufferCreateInfo> for FramebufferConfiguration {
    fn from(value: FramebufferCreateInfo) -> Self {
        FramebufferConfiguration {
            resolution: value.extent,
            attachments: value.attachments,
            layers: value.layers,
        }
    }
}



type VkFramebuffer = vulkano::render_pass::Framebuffer;

#[derive(Clone, Debug)]
pub struct Framebuffer {
    framebuffer: Arc<VkFramebuffer>,
}

impl Framebuffer {
    pub fn new(render_pass: Arc<VkRenderPass>, config: FramebufferConfiguration) -> Result<Self> {

        let create_info = FramebufferCreateInfo{
            attachments: config.attachments,
            extent: config.resolution,
            layers: config.layers,
            ..Default::default()
        };

        let framebuffer = VkFramebuffer::new(render_pass, create_info)
            .inspect_err(|err| error!("Failed to create Framebuffer: {err}"))?;

        Ok(Framebuffer{
            framebuffer,
        })
    }

    pub fn get(&self) -> &Arc<VkFramebuffer> {
        let a = self.framebuffer.handle();
        &self.framebuffer
    }

    pub fn get_resolution(&self) -> [u32; 2] {
        self.framebuffer.extent()
    }

    pub fn get_width(&self) -> u32 {
        self.framebuffer.extent()[0]
    }

    pub fn get_height(&self) -> u32 {
        self.framebuffer.extent()[1]
    }
}

impl From<Arc<VkFramebuffer>> for Framebuffer {
    fn from(value: Arc<VkFramebuffer>) -> Self {
        Framebuffer{
            framebuffer: value,
        }
    }
}