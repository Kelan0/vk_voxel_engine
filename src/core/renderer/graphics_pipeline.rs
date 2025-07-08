use anyhow::{anyhow, Result};
use foldhash::HashMap;
use foldhash::HashSet;
use smallvec::smallvec;
use std::sync::Arc;
use vulkano::device::Device;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::tessellation::TessellationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, PipelineCreateFlags, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::ShaderModule;

type VkGraphicsPipeline = vulkano::pipeline::graphics::GraphicsPipeline;

#[derive(Clone)]
pub struct GraphicsPipelineConfiguration {
    pub flags: PipelineCreateFlags,
    pub subpass_type: PipelineSubpassType,
    pub shader_modules: Vec<Arc<ShaderModule>>,
    pub entry_points: HashMap<u32, &'static str>,
    pub vertex_module_index: u32,
    pub input_assembly_state: InputAssemblyState,
    pub tessellation_state: TessellationState,
    pub viewport_state: ViewportState,
    pub rasterization_state: RasterizationState,
    pub multisample_state: MultisampleState,
    pub color_blend_state: ColorBlendState,
    pub dynamic_states: Vec<DynamicState>,
    pub base_pipeline: Option<DerivativeGraphicsPipeline>,
}

#[derive(Clone)]
enum DerivativeGraphicsPipeline {
    Internal(Arc<VkGraphicsPipeline>),
    Wrapper(GraphicsPipeline)
}

impl GraphicsPipelineConfiguration {
    pub fn new(subpass_type: PipelineSubpassType, shader_modules: Vec<Arc<ShaderModule>>, entry_points: HashMap<u32, &'static str>, vertex_module_index: u32) -> Self {
        let flags = PipelineCreateFlags::empty();
        
        let input_assembly_state = InputAssemblyState{
            topology: PrimitiveTopology::TriangleList,
            ..Default::default()
        };

        let tessellation_state = TessellationState{
            ..Default::default()
        };

        let viewport_state = ViewportState{
            ..Default::default()
        };

        let rasterization_state = RasterizationState{
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::Back,
            front_face: FrontFace::CounterClockwise,
            ..Default::default()
        };

        let multisample_state = MultisampleState{
            ..Default::default()
        };

        let color_blend_state = ColorBlendState{
            attachments: vec![ColorBlendAttachmentState::default()],
            ..Default::default()
        };

        let dynamic_states = vec![];
        
        let base_pipeline = None;

        GraphicsPipelineConfiguration{
            flags,
            subpass_type,
            shader_modules,
            entry_points,
            vertex_module_index,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            rasterization_state,
            multisample_state,
            color_blend_state,
            dynamic_states,
            base_pipeline,
        }
    }
}


#[derive(Clone)]
pub struct GraphicsPipeline {
    pipeline: Arc<VkGraphicsPipeline>
}

impl GraphicsPipeline {
    pub fn new<V: Vertex>(device: Arc<Device>, config: GraphicsPipelineConfiguration) -> Result<Self> {

        let mut stages = smallvec![];

        let mut vertex_module_index = None;
        for (module_index, entry_point_name) in config.entry_points {
            let module = &config.shader_modules[module_index as usize];
            let entry_point = module.entry_point(entry_point_name)
                .ok_or_else(|| anyhow!("Unable to find entry point \"{}\" for shader module at index {}: {:?}", entry_point_name, module_index, module))?;
            let stage = PipelineShaderStageCreateInfo::new(entry_point);
            if config.vertex_module_index == module_index {
                vertex_module_index = Some(stages.len());
            }
            stages.push(stage);
        }

        let create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages).into_pipeline_layout_create_info(device.clone())?;

        if vertex_module_index.is_none() {
            return Err(anyhow!("config.vertex_module_index was not a valid index"));
        }

        let vertex_module_index = vertex_module_index.unwrap();
        
        let vertex_stage: &PipelineShaderStageCreateInfo = &stages[vertex_module_index];
        let vertex_input_state = V::per_vertex().definition(&vertex_stage.entry_point)?;

        let dynamic_state = HashSet::from_iter(config.dynamic_states);

        let pipeline_layout = PipelineLayout::new(device.clone(), create_info)?;

        let mut flags = config.flags;
        
        let base_pipeline = 
            if let Some(DerivativeGraphicsPipeline::Internal(base_pipeline)) = config.base_pipeline {
                Some(base_pipeline)
            } else if let Some(DerivativeGraphicsPipeline::Wrapper(base_pipeline)) = config.base_pipeline {
                Some(base_pipeline.get())
            } else {
                None
            };
        
        if base_pipeline.is_some() {
            flags |= PipelineCreateFlags::DERIVATIVE;
        }
        
        let create_info = GraphicsPipelineCreateInfo{
            flags: config.flags,
            stages,
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(config.input_assembly_state),
            // tessellation_state: Some(tessellation_state),
            viewport_state: Some(config.viewport_state),
            rasterization_state: Some(config.rasterization_state),
            multisample_state: Some(config.multisample_state),
            depth_stencil_state: None,
            color_blend_state: Some(config.color_blend_state),
            dynamic_state,
            subpass: Some(config.subpass_type),
            base_pipeline,
            ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
        };

        let pipeline = VkGraphicsPipeline::new(device.clone(), None, create_info)?;

        Ok(GraphicsPipeline{
            pipeline
        })
    }

    pub fn get(&self) -> Arc<VkGraphicsPipeline> {
        self.pipeline.clone()
    }
}