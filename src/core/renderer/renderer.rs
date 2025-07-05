use crate::application::window::SdlWindow;
use crate::core::renderer::framebuffer::{Framebuffer, FramebufferConfiguration};
use crate::core::renderer::image_view::{ImageView, ImageViewConfiguration};
use crate::core::renderer::render_pass::RenderPass;
use crate::core::renderer::render_pass::RenderPassConfiguration;
use crate::log_error_and_anyhow;
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use smallvec::smallvec;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageViewType;
use vulkano::image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage, SampleCount, SampleCounts};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::MemoryHeapFlags;
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, RenderPassCreateInfo, SubpassDependency, SubpassDescription};
use vulkano::swapchain::{ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, SurfaceInfo, Swapchain, SwapchainCreateInfo};
use vulkano::sync::{AccessFlags, PipelineStages, Sharing};
use vulkano::{DeviceSize, VulkanLibrary};

const ENABLE_VALIDATION_LAYERS: bool = true;

pub struct Renderer {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    queue_details: QueueDetails,
    debug_messenger: Option<DebugUtilsMessenger>,
    device: Arc<Device>,
    queues: HashMap<&'static str, Arc<Queue>>,
    preferred_present_mode: PresentMode,
    present_mode: PresentMode,
    color_format: Format,
    color_space: ColorSpace,
    depth_format: Format,
    surface_capabilities: Option<SurfaceCapabilities>,
    update_swapchain_request: Option<UpdateSwapchainRequest>,
    direct_image_present_enabled: bool,
    swapchain_image_sampled: bool,
    swapchain_info: SwapchainInfo,
    render_pass: Option<RenderPass>,
    max_concurrent_frames: usize
}

#[derive(Default)]
struct UpdateSwapchainRequest {
    image_extent: [u32; 2],
}

#[derive(Default)]
pub struct SwapchainInfo {
    swapchain: Option<Arc<Swapchain>>,
    images: Vec<Arc<Image>>,
    image_views: Vec<ImageView>,
    framebuffers: Vec<Framebuffer>,
    current_frame_idx: u32,
    current_image_idx: u32,
    prev_image_idx: u32,
    image_extent: [u32; 2],
    buffer_mode: SwapchainBufferMode,
}

#[derive(Default)]
pub enum SwapchainBufferMode {
    SingleBuffer,
    #[default]
    DoubleBuffer,
    TripleBuffer,
}




pub struct QueueLayout {
    layout: HashMap<String, QueueLayoutDefinition>
}

pub struct QueueLayoutDefinition {
    name: &'static str,
    required_queues: QueueFlags,
    requires_present: bool,
    group: u32,
    priority: f32,
}

impl QueueLayoutDefinition {
    fn new(name: &'static str) -> Self {
        QueueLayoutDefinition{
            name,
            required_queues: QueueFlags::empty(),
            requires_present: false,
            group: 0,
            priority: 0.0,
        }
    }

    fn requires_queue(mut self, queue: QueueFlags) -> Self {
        self.required_queues |= queue;
        self
    }

    fn requires_present(mut self) -> Self {
        self.requires_present = true;
        self
    }

    fn group(mut self, group: u32) -> Self {
        self.group = group;
        self
    }

    fn priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }
}



pub enum QueueId {
    GraphicsMain,
    ComputeMain,
    TransferMain,
    GraphicsTransferMain,
}

impl QueueId {
    pub fn name(&self) -> &'static str {
        match self {
            QueueId::GraphicsMain => "graphics_main",
            QueueId::ComputeMain => "compute_main",
            QueueId::TransferMain => "transfer_main",
            QueueId::GraphicsTransferMain => "graphics_transfer_main",
        }
    }
}




impl Renderer {
    pub fn new(sdl_window: &SdlWindow) -> Result<Self> {
        info!("Initializing renderer");

        info!("Initializing vulkan");
        let library = VulkanLibrary::new()
            .inspect_err(|_| error!("Error initializing VulkanLLibrary for renderer"))?;

        let instance = Self::create_instance(&library, sdl_window)
            .inspect_err(|_| error!("Error creating Vulkan instance"))?;

        let debug_messenger = if ENABLE_VALIDATION_LAYERS {
            Some(Self::create_debug_utils_messenger(&instance)?)
        } else {
            None
        };

        let surface = Self::create_surface(&instance, sdl_window)
            .inspect_err(|_| error!("Error creating surface for renderer"))?;

        let physical_device = Self::select_physical_device(&instance)
            .inspect_err(|_| error!("Error selecting physical device for renderer"))?;

        let required_queue_flags = QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER;
        let queue_details   = Self::select_queue_families(&physical_device, required_queue_flags, Some(surface.as_ref()))
            .inspect_err(|_| error!("Error selecting queue families for renderer"))?;

        let device_extensions = DeviceExtensions{
            khr_swapchain: true,
            ext_extended_dynamic_state: true,
            ext_extended_dynamic_state2: true,
            ext_extended_dynamic_state3: true,
            ext_line_rasterization: true,
            ..Default::default()
        };

        let device_features = DeviceFeatures{
            fill_mode_non_solid: true,
            shader_sampled_image_array_dynamic_indexing: true,
            shader_uniform_buffer_array_dynamic_indexing: true,
            shader_storage_image_array_dynamic_indexing: true,
            shader_storage_buffer_array_dynamic_indexing: true,
            ..Default::default()
        };

        let queue_layout = [
            QueueLayoutDefinition::new(QueueId::GraphicsMain.name()).requires_queue(QueueFlags::GRAPHICS).priority(1.0).requires_present(),
            QueueLayoutDefinition::new(QueueId::ComputeMain.name()).requires_queue(QueueFlags::COMPUTE).priority(0.5),
            QueueLayoutDefinition::new(QueueId::TransferMain.name()).requires_queue(QueueFlags::TRANSFER).priority(0.2),
            QueueLayoutDefinition::new(QueueId::GraphicsTransferMain.name()).requires_queue(QueueFlags::GRAPHICS | QueueFlags::TRANSFER).priority(0.5),
        ];
        let (device, queues) = Self::create_logical_device(&physical_device, device_extensions, device_features, &queue_details, &queue_layout)
            .inspect_err(|_| error!("Error creating logical device for renderer"))?;

        let direct_image_present_enabled = false;
        let swapchain_image_sampled = false;
        let (width, height) = sdl_window.size_in_pixels();

        let mut renderer = Renderer{
            instance,
            surface,
            physical_device,
            queue_details,
            debug_messenger,
            device,
            queues,
            preferred_present_mode: PresentMode::Immediate,
            present_mode: PresentMode::Immediate,
            color_format: Format::UNDEFINED,
            color_space: ColorSpace::SrgbNonLinear,
            depth_format: Format::UNDEFINED,
            surface_capabilities: None,
            update_swapchain_request: None,
            direct_image_present_enabled,
            swapchain_image_sampled,
            swapchain_info: SwapchainInfo::default(),
            render_pass: None,
            max_concurrent_frames: 3,
        };

        renderer.set_resolution(width, height)?;

        Ok(renderer)
    }

    pub fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn create_instance(library: &Arc<VulkanLibrary>, sdl_window: &SdlWindow) -> Result<Arc<Instance>> {
        info!("Creating Vulkan instance");

        let required_extensions = sdl_window.vulkan_instance_extensions()
            .inspect_err(|err| error!("Failed to retrieve required Vulkan instance extensions from SDL window: {err}"))?;

        let mut enabled_extensions = InstanceExtensions::from_iter(required_extensions.iter().map(|s| s.as_str()));

        info!("Required instance extensions for the supplied window: {enabled_extensions:?}", );

        if ENABLE_VALIDATION_LAYERS {
            warn!("Enabling Vulkan validation layers - This will probably heavily impact performance");
            enabled_extensions.ext_debug_report = true;
            enabled_extensions.ext_debug_utils = true;
        }

        let enabled_layers = Self::select_validation_layers(library)?;

        let instance = Instance::new(library.clone(), InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions,
            enabled_layers,
            ..Default::default()
        })?;

        Ok(instance)
    }

    fn select_validation_layers(library: &Arc<VulkanLibrary>) -> Result<Vec<String>> {
        let required_validation_layers = vec!["VK_LAYER_KHRONOS_validation"];

        let mut available_layer_properties = library.layer_properties()
            .inspect_err(|err| error!("Unable to enumerate Vulkan instance layer properties: {err}"))?;

        let mut layer_names = vec![];

        for validation_layer_name in required_validation_layers {
            let found = available_layer_properties.any(|layer| validation_layer_name == layer.name());

            if !found {
                return Err(log_error_and_anyhow!("Required validation layer \"{validation_layer_name}\" was not found"));
            }

            layer_names.push(validation_layer_name.to_string());
        }

        Ok(layer_names)
    }

    fn create_surface(instance: &Arc<Instance>, sdl_window: &SdlWindow) -> Result<Arc<Surface>> {
        info!("Creating Vulkan SDL surface");

        let surface =
            unsafe { Surface::from_window_ref(instance.clone(), &sdl_window) }
            .inspect_err(|err| error!("Failed to construct Vulkan surface from provided SDL window handle - reason: {err}"))?;

        Ok(surface)
    }

    fn select_physical_device(instance: &Arc<Instance>) -> Result<Arc<PhysicalDevice>> {
        info!("Selecting physical device for rendering");

        let all_physical_devices: Vec<_> = instance.enumerate_physical_devices()
            .inspect_err(|err| error!("Failed to enumerate physical devices: {err}"))?
            .collect();

        if all_physical_devices.is_empty() {
            return Err(log_error_and_anyhow!("No physical devices available"));
        }

        for physical_device in &all_physical_devices {
            let device_name = &physical_device.properties().device_name;
            let suitable = Self::is_physical_device_suitable(physical_device);
            info!("Found physical device: \"{device_name}\"{}", if !suitable { " (Not suitable)" } else { "" });
        }

        let mut all_physical_devices: Vec<_> = all_physical_devices.into_iter()
            .filter(Self::is_physical_device_suitable)
            .collect();

        all_physical_devices.sort_by(Self::compare_physical_devices);

        let selected_physical_device = all_physical_devices.pop();

        let selected_physical_device = selected_physical_device.ok_or_else(|| {
            log_error_and_anyhow!("Failed to select a usable physical device")
        })?;

        info!("Selecting physical device: {}", selected_physical_device.properties().device_name);

        for family in selected_physical_device.queue_family_properties() {
            info!("Found a queue family with {:?} queue(s) for [{:?}]", family.queue_count, family.queue_flags);
        }
        Ok(selected_physical_device)
    }

    //noinspection DuplicatedCode
    // Return Greater if physical_device1 is better than physical_device2
    // Return Less if physical_device1 is worse than physical_device2
    // Return Equal otherwise
    fn compare_physical_devices(physical_device1: &Arc<PhysicalDevice>, physical_device2: &Arc<PhysicalDevice>) -> Ordering {
        let props1 = physical_device1.properties();
        let props2 = physical_device2.properties();

        if props1.device_type != props2.device_type {
            if props1.device_type == PhysicalDeviceType::DiscreteGpu {
                return Greater; // device1 is better than device2
            }
            if props2.device_type == PhysicalDeviceType::DiscreteGpu {
                return Less; // device2 is better than device1
            }
        }

        let features1 = physical_device1.supported_features();
        let features2 = physical_device2.supported_features();

        let num_features_1 = features1.into_iter().filter(|&(_feature_name, feature_enabled)|{ feature_enabled }).count();
        let num_features_2 = features2.into_iter().filter(|&(_feature_name, feature_enabled)|{ feature_enabled }).count();

        if num_features_1 > num_features_2 {
            return Greater; // device1 has more features than device2
        }
        if num_features_1 < num_features_2 {
            return Less; // device2 has more features than device1
        }

        let mem_props1 = physical_device1.memory_properties();
        let mem_props2 = physical_device2.memory_properties();

        let vram1 = mem_props1.memory_heaps.iter()
            .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<DeviceSize>();
        let vram2 = mem_props2.memory_heaps.iter()
            .filter(|heap| heap.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<DeviceSize>();

        if vram1 > vram2 {
            return Greater; // device1 has more memory than device2
        }
        if vram2 > vram1 {
            return Less; // device2 has more memory than device1
        }

        // TODO: compare more stuff

        Equal
    }

    fn is_physical_device_suitable(physical_device: &Arc<PhysicalDevice>) -> bool {
        let props = physical_device.properties();

        matches!(props.device_type, PhysicalDeviceType::DiscreteGpu | PhysicalDeviceType::IntegratedGpu | PhysicalDeviceType::VirtualGpu)
    }

    fn select_queue_families(physical_device: &Arc<PhysicalDevice>, required_queue_flags: QueueFlags, surface: Option<&Surface>) -> Result<QueueDetails> {

        let requires_graphics = required_queue_flags.contains(QueueFlags::GRAPHICS);
        let requires_compute = required_queue_flags.contains(QueueFlags::COMPUTE);
        let requires_transfer = required_queue_flags.contains(QueueFlags::TRANSFER);
        let requires_sparse_binding = required_queue_flags.contains(QueueFlags::SPARSE_BINDING);
        let requires_protected = required_queue_flags.contains(QueueFlags::PROTECTED);
        let requires_present = surface.is_some();

        let mut queue_details = QueueDetails::default();

        let queue_family_properties = physical_device.queue_family_properties();
        for (i, prop) in queue_family_properties.iter().enumerate() {
            let supports_graphics = prop.queue_flags.contains(QueueFlags::GRAPHICS);
            let supports_compute = prop.queue_flags.contains(QueueFlags::COMPUTE);
            let supports_transfer = prop.queue_flags.contains(QueueFlags::TRANSFER);
            let supports_sparse_binding = prop.queue_flags.contains(QueueFlags::SPARSE_BINDING);
            let supports_protected = prop.queue_flags.contains(QueueFlags::PROTECTED);
            let supports_present = if let Some(surface) = surface {
                physical_device.surface_support(i as u32, surface).unwrap_or(false)
            } else {
                false
            };

            if supports_graphics { queue_details.graphics_queue_family_index.get_or_insert(i as u32); }
            if supports_compute { queue_details.compute_queue_family_index.get_or_insert(i as u32); }
            if supports_transfer { queue_details.transfer_queue_family_index.get_or_insert(i as u32); }
            if supports_sparse_binding { queue_details.sparse_binding_queue_family_index.get_or_insert(i as u32); }
            if supports_protected { queue_details.protected_queue_family_index.get_or_insert(i as u32); }
            if supports_present { queue_details.present_queue_family_index.get_or_insert(i as u32); }
        }

        let mut missing_queue_types = QueueFlags::empty();
        if requires_graphics && queue_details.graphics_queue_family_index.is_none() { missing_queue_types |= QueueFlags::GRAPHICS; }
        if requires_compute && queue_details.compute_queue_family_index.is_none() { missing_queue_types |= QueueFlags::COMPUTE; }
        if requires_transfer && queue_details.transfer_queue_family_index.is_none() { missing_queue_types |= QueueFlags::TRANSFER; }
        if requires_sparse_binding && queue_details.sparse_binding_queue_family_index.is_none() { missing_queue_types |= QueueFlags::SPARSE_BINDING; }
        if requires_protected && queue_details.protected_queue_family_index.is_none() { missing_queue_types |= QueueFlags::PROTECTED; }
        let missing_present = requires_present && queue_details.present_queue_family_index.is_none();


        if !missing_queue_types.is_empty() {
            return Err(log_error_and_anyhow!("Failed to select queue families - Required support for queues: [{:?}], missing support for queues: [{:?}]", required_queue_flags, missing_queue_types));
        }

        if requires_present && missing_present {
            return Err(log_error_and_anyhow!("Failed to select queue families - Required support for PRESENT but no queues supported it"));
        }

        Ok(queue_details)
    }

    fn create_logical_device(physical_device: &Arc<PhysicalDevice>, enabled_extensions: DeviceExtensions, enabled_features: DeviceFeatures, queue_details: &QueueDetails, queue_layout: &[QueueLayoutDefinition])
        -> Result<(Arc<Device>, HashMap<&'static str, Arc<Queue>>)> {

        info!("Creating logical device for Vulkan");

        let mut queue_create_infos = vec![];
        let enabled_extensions = DeviceExtensions{ ..enabled_extensions };
        let enabled_features = DeviceFeatures { ..enabled_features };

        let queue_family_properties = physical_device.queue_family_properties();

        // Flatten the indices in queue_details into a set of unique indices
        let unique_queue_indices: HashSet<u32> = queue_details.indices().into_iter().flatten().collect();

        // Convert the provided list of QueueLayoutDefinition into a HashMap
        let mut queue_layout: HashMap<&str, (&QueueLayoutDefinition, bool)> = queue_layout.iter().map(|e| (e.name, (e, false))).collect();

        let mut queue_map: HashMap<(u32, u32), &QueueLayoutDefinition> = HashMap::new();

        for queue_family_index in unique_queue_indices {
            let mut queue_priorities = vec![];

            if queue_layout.is_empty() {
                break;
            }
            let props = &queue_family_properties[queue_family_index as usize];

            let is_present = queue_details.present_queue_family_index == Some(queue_family_index);

            for (_id, (layout_def, is_removed)) in queue_layout.iter_mut() {

                if queue_priorities.len() >= props.queue_count as usize {
                    break;
                }
                if *is_removed {
                    continue;
                }

                if props.queue_flags.contains(layout_def.required_queues) && (!layout_def.requires_present || is_present) {
                    queue_map.insert((queue_family_index, queue_priorities.len() as u32), layout_def);
                    queue_priorities.push(layout_def.priority);
                    *is_removed = true;
                }
            }

            queue_layout.retain(|_, (_, is_removed)| !*is_removed );

            if queue_priorities.is_empty() {
                error!("Could not initialize the desired queue layout for queue family {queue_family_index}");
                continue;
            }

            let queue_create_info = QueueCreateInfo{
                queue_family_index,
                queues: queue_priorities.clone(),
                ..Default::default()
            };
            queue_create_infos.push(queue_create_info);
        }


        let create_info = DeviceCreateInfo{
            queue_create_infos,
            enabled_extensions,
            enabled_features,
            ..Default::default()
        };
        let (device, queues_iter) = Device::new(physical_device.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan device: {err}"))?;

        let mut queues = HashMap::new();
        for queue in queues_iter {
            let layout_def = queue_map.get(&(queue.queue_family_index(), queue.queue_index())).unwrap();
            queues.insert(layout_def.name, queue);
        }

        Ok((device, queues))
    }

    fn create_debug_utils_messenger(instance: &Arc<Instance>) -> Result<DebugUtilsMessenger> {
        let mut create_info = DebugUtilsMessengerCreateInfo::user_callback(unsafe {
            DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                if message_severity.contains(DebugUtilsMessageSeverity::ERROR) {
                    error!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::WARNING) {
                    warn!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::INFO) {
                    info!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
                if message_severity.contains(DebugUtilsMessageSeverity::VERBOSE) {
                    debug!("[VULKAN API][{:?}][{:?}] {}", message_severity, message_type, callback_data.message);
                }
            })
        });

        create_info.message_severity = DebugUtilsMessageSeverity::INFO | DebugUtilsMessageSeverity::WARNING | DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::VERBOSE;
        create_info.message_type = DebugUtilsMessageType::GENERAL | DebugUtilsMessageType::VALIDATION | DebugUtilsMessageType::PERFORMANCE;

        let messenger = DebugUtilsMessenger::new(instance.clone(), create_info)?;
        Ok(messenger)
    }

    fn create_render_pass(&mut self) -> Result<()> {
        info!("Creating render pass");

        let color_attachment = AttachmentDescription{
            format: self.color_format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::DontCare,
            store_op: AttachmentStoreOp::Store,
            stencil_load_op: Some(AttachmentLoadOp::DontCare),
            stencil_store_op: Some(AttachmentStoreOp::DontCare),
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::PresentSrc,
            ..Default::default()
        };

        let subpass_description = SubpassDescription{
            color_attachments: vec![
                Some(AttachmentReference{attachment: 0, layout: ImageLayout::ColorAttachmentOptimal, ..Default::default()})
            ],
            ..Default::default()
        };

        let subpass_dependency = SubpassDependency{
            src_subpass: None,
            dst_subpass: Some(0),
            src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            src_access: AccessFlags::empty(),
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let create_info = RenderPassConfiguration::new()
            .add_attachment(color_attachment)
            .add_subpass(subpass_description)
            .add_subpass_dependency(subpass_dependency);

        let render_pass = RenderPass::new(self.device.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan RenderPass: {err}"))?;

        self.render_pass = Some(render_pass);

        Ok(())
    }

    fn get_surface_capabilities(physical_device: &PhysicalDevice, surface: &Surface, surface_info: SurfaceInfo) -> Result<SurfaceCapabilities> {

        let surface_capabilities = physical_device.surface_capabilities(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get surface capabilities for the Vulkan physical device: {err}"))?;

        Ok(surface_capabilities)
    }

    fn init_surface_details(&mut self, width: u32, height: u32) -> Result<bool> {
        info!("Reinitializing renderer surface for resolution [{width} x {height}]");

        let surface = self.surface.as_ref();
        let surface_info = SurfaceInfo::default();

        let surface_capabilities = Self::get_surface_capabilities(&self.physical_device, &self.surface, surface_info.clone())?;

        self.swapchain_info.image_extent[0] = u32::clamp(width, surface_capabilities.min_image_extent[0], surface_capabilities.max_image_extent[0]);
        self.swapchain_info.image_extent[1] = u32::clamp(height, surface_capabilities.min_image_extent[1], surface_capabilities.max_image_extent[1]);

        self.surface_capabilities = Some(surface_capabilities);

        let formats = self.physical_device.surface_formats(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get surface formats for the Vulkan physical device: {err}"))?;

        if formats.is_empty() {
            error!("Current Vulkan device supports no surface formats for this surface");
            return Ok(false);
        }

        let use_srgb = false;

        let mut selected_format_idx = usize::MAX;

        for (i, (format, color_space)) in formats.iter().enumerate() {
            if use_srgb {
                if *format == Format::B8G8R8A8_SRGB && *color_space == ColorSpace::SrgbNonLinear {
                    selected_format_idx = i;
                }
            } else if *format == Format::B8G8R8A8_UNORM && *color_space == ColorSpace::SrgbNonLinear {
                selected_format_idx = i;
            }
        }
        if selected_format_idx == usize::MAX {
            warn!("Preferred surface format and colour space was not found. Defaulting to first available option");
            selected_format_idx = 0;
        }

        (self.color_format, self.color_space) = formats[selected_format_idx];

        info!("Using output render colour format {:?} with colour space {:?}", self.color_format, self.color_space);

        let present_modes = self.physical_device.surface_present_modes(surface, surface_info.clone())
            .inspect_err(|err| error!("Failed to get present modes for Vulkan physical device: {err}"))?;

        if present_modes.is_empty() {
            error!("Current Vulkan device supports no present modes for this surface");
            return Ok(false);
        }

        let present_mode;

        if let Some(val) = present_modes.iter().find(|val| **val == self.preferred_present_mode) {
            present_mode = val;
        } else {
            present_mode = &present_modes[0];
            warn!("Preferred surface present mode {:?} was not found. Defaulting to {:?}", self.preferred_present_mode, present_mode);
        }

        self.present_mode = *present_mode;
        self.depth_format = Format::D32_SFLOAT;

        Ok(true)
    }

    fn create_swapchain_image_views(&mut self) -> Result<()> {
        self.swapchain_info.image_views = vec![];

        for image in &self.swapchain_info.images {
            let config = ImageViewConfiguration::new()
                .set_view_type(ImageViewType::Dim2d)
                .set_format(self.color_format)
                .set_subresource_range(ImageSubresourceRange{
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..1
                });

            let image_view = ImageView::new(image.clone(), config)?;
            self.swapchain_info.image_views.push(image_view);
        }

        Ok(())
    }

    fn create_swapchain_framebuffers(&mut self) -> Result<()> {
        let render_pass = self.render_pass.as_ref().unwrap().get();

        self.swapchain_info.framebuffers.clear();

        for image_view in &self.swapchain_info.image_views {

            let config = FramebufferConfiguration::new()
                .set_resolution(self.swapchain_info.image_extent)
                .add_attachment(image_view.get().clone());

            let framebuffer = Framebuffer::new(render_pass.clone(), config)?;
            self.swapchain_info.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> Result<()> {
        info!("Recreating Vulkan swapchain...");

        let update_swapchain_request = self.update_swapchain_request.take().unwrap();

        unsafe {
            self.device.wait_idle()?;
        }

        let success = self.init_surface_details(update_swapchain_request.image_extent[0], update_swapchain_request.image_extent[1])?;
        if !success {
            error!("Error when initializing surface details. Unable to recreate swapchain");
            return Ok(());
        }

        self.create_render_pass()
            .inspect_err(|_| error!("Failed to create RenderPass"))?;

        self.swapchain_info.framebuffers.clear();
        self.swapchain_info.image_views.clear();
        self.swapchain_info.images.clear();
        self.swapchain_info.swapchain = None;

        let queue_details = &self.queue_details;
        let surface_capabilities = self.surface_capabilities.as_ref().unwrap();

        let mut image_count: u32 = match self.swapchain_info.buffer_mode {
            SwapchainBufferMode::SingleBuffer => 1,
            SwapchainBufferMode::DoubleBuffer => 2,
            SwapchainBufferMode::TripleBuffer => 3,
        };

        image_count = u32::max(image_count, surface_capabilities.min_image_count);
        if let Some(max_image_count) = surface_capabilities.max_image_count {
            image_count = u32::min(image_count, max_image_count);
        }

        let image_array_layers: u32 = 1;

        let mut image_usage = ImageUsage::COLOR_ATTACHMENT;
        if self.direct_image_present_enabled {
            image_usage |= ImageUsage::TRANSFER_DST;
        }
        if self.swapchain_image_sampled {
            image_usage |= ImageUsage::SAMPLED;
        }

        let image_sharing =
            if queue_details.graphics_queue_family_index != queue_details.present_queue_family_index {
                warn!("The GRAPHICS queue family is not the same as the PRESENT queue family - This may affect performance");
                let queues = smallvec![queue_details.graphics_queue_family_index.unwrap(), queue_details.present_queue_family_index.unwrap()];
                Sharing::Concurrent(queues)
            } else {
                Sharing::Exclusive
            };

        let create_info = SwapchainCreateInfo{
            min_image_count: image_count,
            image_format: self.color_format,
            image_color_space: self.color_space,
            image_extent: self.swapchain_info.image_extent,
            image_array_layers,
            image_usage,
            image_sharing,

            pre_transform: surface_capabilities.current_transform,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: self.present_mode,

            ..Default::default()
        };

        let (swapchain, images) = Swapchain::new(self.device.clone(), self.surface.clone(), create_info)
            .inspect_err(|err| error!("Failed to create Vulkan swapchain: {err}"))?;

        self.swapchain_info.swapchain = Some(swapchain);
        self.swapchain_info.images = images;

        self.create_swapchain_image_views()
            .inspect_err(|err| error!("Failed to create ImageViews for the swapchain: {err}"))?;

        self.create_swapchain_framebuffers()
            .inspect_err(|err| error!("Failed to create Framebuffers for the swapchain: {err}"))?;

        info!("Created swapchain for resolution [{} x {}]", self.get_resolution_width(), self.get_resolution_height());

        Ok(())
    }

    pub fn get_render_pass(&self) -> &RenderPass {
        self.render_pass.as_ref().unwrap()
    }

    pub fn set_resolution(&mut self, width: u32, height: u32) -> Result<bool> {
        if width == self.swapchain_info.image_extent[0] && height == self.swapchain_info.image_extent[1] {
            return Ok(false); // width & height is unchanged
        }

        if self.update_swapchain_request.is_none() {
            self.update_swapchain_request = Some(Default::default());
        }
        let a = self.update_swapchain_request.as_mut().unwrap();
        a.image_extent[0] = width;
        a.image_extent[1] = height;

        Ok(true)
    }

    pub fn get_resolution_width(&self) -> u32 {
        self.swapchain_info.image_extent[0]
    }

    pub fn get_resolution_height(&self) -> u32 {
        self.swapchain_info.image_extent[1]
    }

    pub fn get_color_format(&self) -> Format {
        self.color_format
    }

    pub fn get_color_space(&self) -> ColorSpace {
        self.color_space
    }

    pub fn pre_render(&mut self) -> Result<()> {
        if self.update_swapchain_request.is_some() {
            self.recreate_swapchain()?;
        }

        Ok(())
    }
}




#[derive(Default, Debug)]
struct QueueDetails {
    graphics_queue_family_index: Option<u32>,
    compute_queue_family_index: Option<u32>,
    transfer_queue_family_index: Option<u32>,
    sparse_binding_queue_family_index: Option<u32>,
    protected_queue_family_index: Option<u32>,
    present_queue_family_index: Option<u32>,
}

impl QueueDetails {
    fn indices(&self) -> [Option<u32>; 6] {
        [
            self.graphics_queue_family_index,
            self.compute_queue_family_index,
            self.transfer_queue_family_index,
            self.sparse_binding_queue_family_index,
            self.protected_queue_family_index,
            self.present_queue_family_index,
        ]
    }
}
