use anyhow::Result;
use log::error;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::format::ClearValue;
use vulkano::image::ImageLayout;
use vulkano::render_pass::{AttachmentDescription, AttachmentReference, RenderPassCreateInfo, SubpassDependency, SubpassDescription};

#[derive(Default, Clone, Debug)]
pub struct RenderPassConfiguration {
    pub attachments: Vec<AttachmentDescription>,
    pub subpass_dependencies: Vec<SubpassDependency>,
    pub subpass_descriptions: Vec<SubpassDescription>,
    pub clear_values: Vec<ClearValue>,
}

impl RenderPassConfiguration {
    pub fn new() -> Self {
        Default::default()
    }
    
    pub fn add_attachment(mut self, attachment_description: AttachmentDescription) -> Self {
        self.attachments.push(attachment_description);
        if self.clear_values.len() < self.attachments.len() {
            self.clear_values.resize(self.attachments.len(), ClearValue::Float([0.0, 0.0, 0.0, 0.0]))
        }
        self
    }

    pub fn set_attachments(mut self, attachment_descriptions: &[AttachmentDescription]) -> Self {
        self.attachments.clear();
        for attachment in attachment_descriptions {
            self = self.add_attachment(attachment.clone());
        }
        self
    }

    pub fn add_subpass(mut self, subpass_configuration: SubpassDescription) -> Self {
        self.subpass_descriptions.push(subpass_configuration);
        self
    }

    pub fn set_subpasses(mut self, subpass_configurations: &[SubpassDescription]) -> Self {
        self.subpass_descriptions.clear();
        for subpass_configuration in subpass_configurations {
            self = self.add_subpass(subpass_configuration.clone());
        }
        self
    }

    pub fn add_subpass_dependency(mut self, subpass_dependency: SubpassDependency) -> Self {
        self.subpass_dependencies.push(subpass_dependency);
        self
    }

    pub fn set_subpass_dependencies(mut self, subpass_dependencies: &[SubpassDependency]) -> Self {
        self.subpass_dependencies.clear();
        for subpass_dependency in subpass_dependencies {
            self = self.add_subpass_dependency(subpass_dependency.clone());
        }
        self
    }

    pub fn set_clear_values(mut self, clear_values: &[ClearValue]) -> Self {
        self.clear_values.clear();
        for clear_value in clear_values {
            self.clear_values.push(clear_value.clone());
        }
        if self.clear_values.len() < self.attachments.len() {
            self.clear_values.resize(self.attachments.len(), ClearValue::Float([0.0, 0.0, 0.0, 0.0]));
        }
        self
    }

    pub fn set_clear_value(mut self, attachment: u32, clear_value: ClearValue) -> Self {
        let idx = attachment as usize;
        if self.clear_values.len() < idx {
            self.clear_values.resize(idx, ClearValue::Float([0.0, 0.0, 0.0, 0.0]));
        }
        self.clear_values[idx] = clear_value;
        self
    }
}

#[derive(Default, Clone, Debug)]
pub struct SubpassConfiguration  {
    pub attachment_references: Vec<AttachmentReference>,
    pub color_attachments: Vec<usize>,
    pub input_attachments: Vec<usize>,
    pub preserve_attachments: Vec<usize>,
    pub depth_stencil_attachment: Option<usize>,
}

impl SubpassConfiguration {
    pub fn new() -> Self {
        Default::default()
    }
    
    pub fn add_colour_attachment_ref(&mut self, attachment_reference: AttachmentReference) -> &Self {
        self.color_attachments.push(self.attachment_references.len());
        self.attachment_references.push(attachment_reference);
        self
    }

    pub fn add_colour_attachment(&mut self, attachment: u32, layout: ImageLayout) -> &Self {
        self.add_colour_attachment_ref(AttachmentReference{ attachment, layout, ..Default::default() })
    }

    pub fn add_input_attachment_ref(&mut self, attachment_reference: AttachmentReference) -> &Self {
        self.input_attachments.push(self.attachment_references.len());
        self.attachment_references.push(attachment_reference);
        self
    }

    pub fn add_input_attachment(&mut self, attachment: u32, layout: ImageLayout) -> &Self {
        self.add_input_attachment_ref(AttachmentReference{ attachment, layout, ..Default::default() })
    }

    pub fn add_preserve_attachment_ref(&mut self, attachment_reference: AttachmentReference) -> &Self {
        self.preserve_attachments.push(self.attachment_references.len());
        self.attachment_references.push(attachment_reference);
        self
    }

    pub fn add_preserve_attachment(&mut self, attachment: u32, layout: ImageLayout) -> &Self {
        self.add_preserve_attachment_ref(AttachmentReference{ attachment, layout, ..Default::default() })
    }

    pub fn set_depth_stencil_attachment_ref(&mut self, attachment_reference: AttachmentReference) -> &Self {
        self.depth_stencil_attachment = Some(self.attachment_references.len());
        self.attachment_references.push(attachment_reference);
        self
    }

    pub fn set_depth_stencil_attachment(&mut self, attachment: u32, layout: ImageLayout) -> &Self {
        self.set_depth_stencil_attachment_ref(AttachmentReference{ attachment, layout, ..Default::default() })
    }
}


pub type VkRenderPass = vulkano::render_pass::RenderPass;

#[derive(Clone, Debug)]
pub struct RenderPass {
    render_pass: Arc<VkRenderPass>
}

impl RenderPass {
    pub fn new(device: Arc<Device>, config: RenderPassConfiguration) -> Result<Self> {

        let create_info = RenderPassCreateInfo{
            attachments: config.attachments,
            dependencies: config.subpass_dependencies,
            subpasses: config.subpass_descriptions,
            ..Default::default()
        };
        
        let render_pass = VkRenderPass::new(device, create_info)
            .inspect_err(|err| error!("Failed to create RenderPass: {err}"))?;
        
        Ok(RenderPass{
            render_pass
        })
    }
    
    pub fn get(&self) -> &Arc<VkRenderPass> {
        &self.render_pass
    }
}

impl From<Arc<VkRenderPass>> for RenderPass {
    fn from(value: Arc<VkRenderPass>) -> Self {
        RenderPass{
            render_pass: value
        }
    }
}