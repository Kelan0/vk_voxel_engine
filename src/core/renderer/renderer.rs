use crate::application::window::SdlWindow;
use crate::{log_error_and_anyhow, log_error_and_throw};
use anyhow::{anyhow, Result};
use log::{debug, error, info, warn};
use smallvec::smallvec;
use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::mem;
use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage, SampleCount};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::MemoryHeapFlags;
use vulkano::render_pass::{AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateFlags, RenderPassCreateInfo, SubpassDependency, SubpassDescription};
use vulkano::swapchain::{ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, SurfaceInfo, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::{AccessFlags, GpuFuture, PipelineStages, Sharing};
use vulkano::{swapchain, sync, DeviceSize, Validated, VulkanError, VulkanLibrary};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::pool::CommandBufferAllocateInfo;

const ENABLE_VALIDATION_LAYERS: bool = true;

pub type PrimaryCommandBuffer = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;
pub type SecondaryCommandBuffer = AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>;


// #[derive(Debug)]
pub struct Renderer {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queue_details: QueueDetails,
    queues: HashMap<&'static str, Arc<Queue>>,
    // command_pools: HashMap<u32, CommandPool>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    debug_messenger: Option<DebugUtilsMessenger>,
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
    render_pass: Option<Arc<RenderPass>>,
    max_concurrent_frames: u32,
    current_frame_index: u32,
}

#[derive(Default, Debug)]
struct UpdateSwapchainRequest {
    image_extent: [u32; 2],
}

pub struct SwapchainInfo {
    swapchain: Option<Arc<Swapchain>>,
    images: Vec<Arc<Image>>,
    image_views: Vec<Arc<ImageView>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    // command_buffers: Vec<Arc<CommandBuffer>>,
    acquire_future: Option<SwapchainAcquireFuture>,
    in_flight_frames: Vec<Box<dyn GpuFuture>>,
    current_image_idx: u32,
    prev_image_idx: u32,
    image_extent: [u32; 2],
    buffer_mode: SwapchainBufferMode,
}

impl SwapchainInfo {
    fn new(device: &Arc<Device>) -> Self {
        SwapchainInfo{
            swapchain: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            framebuffers: Default::default(),
            acquire_future: Default::default(),
            in_flight_frames: Default::default(),
            current_image_idx: 0,
            prev_image_idx: 0,
            image_extent: Default::default(),
            buffer_mode: Default::default(),
        }
    }
}

#[derive(Default, Debug)]
pub enum SwapchainBufferMode {
    SingleBuffer,
    #[default]
    DoubleBuffer,
    TripleBuffer,
}




#[derive(Debug)]
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



#[derive(Debug)]
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

pub enum BeginFrameResult<E = anyhow::Error> {
    Begin(PrimaryCommandBuffer),
    Skip,
    Err(E)
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

        // let command_pools = Self::create_command_pools(&device, &queue_details)
        //     .inspect_err(|_| error!("Error creating command pools for renderer"))?;
        
        let command_buffer_allocator = Self::create_command_buffer_allocator(&device)
            .inspect_err(|_| error!("Error creating command buffer allocator"))?;

        let direct_image_present_enabled = false;
        let swapchain_image_sampled = false;
        let (width, height) = sdl_window.size_in_pixels();

        // Initial "previous frame" - There is no previous frame on the first frame, so the GpuFuture is just now

        let swapchain_info = SwapchainInfo::new(&device);

        let mut renderer = Renderer{
            instance,
            surface,
            physical_device,
            device,
            queue_details,
            queues,
            // command_pools,
            command_buffer_allocator,
            debug_messenger,
            preferred_present_mode: PresentMode::Immediate,
            present_mode: PresentMode::Immediate,
            color_format: Format::UNDEFINED,
            color_space: ColorSpace::SrgbNonLinear,
            depth_format: Format::UNDEFINED,
            surface_capabilities: None,
            update_swapchain_request: None,
            direct_image_present_enabled,
            swapchain_image_sampled,
            swapchain_info,
            render_pass: None,
            max_concurrent_frames: 3,
            current_frame_index: 0,
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

        let create_info = InstanceCreateInfo{
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions,
            enabled_layers,
            ..Default::default()
        };

        let instance = Instance::new(library.clone(), create_info)?;

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

        let unique_queue_indices = queue_details.unique_indices();

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

    // fn create_command_pools(device: &Arc<Device>, queue_details: &QueueDetails) -> Result<HashMap<u32, CommandPool>> {
    //
    //     let mut command_pools = HashMap::new();
    //
    //     for queue_family_index in queue_details.unique_indices() {
    //         let command_pool = CommandPoolConfiguration::new()
    //             .set_queue_family_index(queue_details.graphics_queue_family_index.unwrap())
    //             .set_transient(false)
    //             .set_reset_command_buffer(true)
    //             .build(device.clone())
    //             .inspect_err(|err| error!("Failed to create CommandPool: {err}"))?;
    //
    //         command_pools.insert(queue_family_index, command_pool);
    //     }
    //
    //     Ok(command_pools)
    // }

    fn create_command_buffer_allocator(device: &Arc<Device>, ) -> Result<Arc<StandardCommandBufferAllocator>> {
        let create_info = StandardCommandBufferAllocatorCreateInfo{
            primary_buffer_count: 32,
            secondary_buffer_count: 8,
            ..Default::default()
        };

        let allocator = StandardCommandBufferAllocator::new(device.clone(), create_info);
        Ok(Arc::new(allocator))

        // CommandBufferAllocator::new(device.clone(), 32, 0)
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
            load_op: AttachmentLoadOp::Clear,
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

        let create_info = RenderPassCreateInfo{
            attachments: vec![color_attachment],
            subpasses: vec![subpass_description],
            dependencies: vec![subpass_dependency],
            ..Default::default()
        };

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
            let image_view = ImageView::new_default(image.clone())?;

            self.swapchain_info.image_views.push(image_view);
        }

        Ok(())
    }

    fn create_swapchain_framebuffers(&mut self) -> Result<()> {
        let render_pass = self.get_render_pass();

        self.swapchain_info.framebuffers.clear();

        for image_view in &self.swapchain_info.image_views {

            let create_info = FramebufferCreateInfo{
                attachments: vec![image_view.clone()],
                extent: self.swapchain_info.image_extent,
                layers: 1,
                ..Default::default()
            };

            let framebuffer = Framebuffer::new(render_pass.clone(), create_info)?;

            self.swapchain_info.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> Result<()> {
        info!("Recreating Vulkan swapchain...");

        let update_swapchain_request = self.update_swapchain_request.take().unwrap();

        self.flush_rendering()?;

        let success = self.init_surface_details(update_swapchain_request.image_extent[0], update_swapchain_request.image_extent[1])?;
        if !success {
            error!("Error when initializing surface details. Unable to recreate swapchain");
            return Ok(());
        }

        self.create_render_pass()
            .inspect_err(|_| error!("Failed to create RenderPass"))?;

        self.swapchain_info.in_flight_frames.clear();
        self.swapchain_info.acquire_future = None;

        // self.swapchain_info.command_buffers.clear();
        // for i in 0..self.max_concurrent_frames {
        //     let cmd_pool = self.get_graphics_command_pool_mut();
        //     _ = cmd_pool.free_command_buffer_if_exists(&format!("swapchain_cmd{i}"))?;
        // }

        if self.swapchain_info.swapchain.is_some() {
            let swapchain = self.swapchain_info.swapchain.as_ref().unwrap();
            let ref_count = Arc::strong_count(swapchain);
            if ref_count > 1 {
                error!("Unable to recreate swapchain - It has {} strong references preventing it from being dropped", ref_count - 1);
            }
        }

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

        // self.swapchain_info.command_buffers.reserve(self.max_concurrent_frames as usize);
        // 
        // for i in 0..self.max_concurrent_frames {
        //     let alloc_config = CommandBufferAllocConfiguration::new(&format!("swapchain_cmd{i}"));
        // 
        //     let cmd_buf = self.get_graphics_command_pool_mut().allocate_command_buffer(CommandBufferLevel::Primary, alloc_config)
        //         .inspect_err(|err| error!("Failed to allocate command buffer {i} for swapchain: {err}"))?;
        // 
        //     self.swapchain_info.command_buffers.insert(i as usize, cmd_buf.clone());
        // }

        self.swapchain_info.in_flight_frames.resize_with(self.max_concurrent_frames as usize, || {
            Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>
        });

        info!("Created swapchain for resolution [{} x {}]", self.get_resolution_width(), self.get_resolution_height());

        Ok(())
    }

    pub fn debug_print_ref_counts(&self) {

        let swapchain_ref_count = self.swapchain_info.swapchain.as_ref().map_or(0, |val| { Arc::strong_count(val) });
        debug!("Device has {} references - Swapchain has {} references", Arc::strong_count(&self.device), swapchain_ref_count);
    }

    pub fn begin_frame(&mut self) -> BeginFrameResult {

        if self.update_swapchain_request.is_some() {
            match self.recreate_swapchain() {
                Err(err) => {
                    error!("Failed to recreate Swapchain");
                    return BeginFrameResult::Err(err);
                },
                _ => {}
            }

            return BeginFrameResult::Skip;
        }

        let allocator = self.command_buffer_allocator.clone();
        let queue_family_index = self.queue_details.graphics_queue_family_index.unwrap();

        let swapchain = match self.get_swapchain() {
            Ok(r) => r,
            Err(err) => return BeginFrameResult::Err(log_error_and_throw!(err, "Failed to get swapchain"))
        };

        // ==== ACQUIRE THE NEXT IMAGE ====

        let a = Instant::now();
        let (image_index, is_suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.request_recreate_swapchain();
                return BeginFrameResult::Skip;
            }
            Err(err) => return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to acquire next image for begin_frame"))
        };

        let b = Instant::now();

        if b.duration_since(a).as_secs_f64() > 0.0001 {
            debug!("acquire_next_image blocked for {:.4} msec", b.duration_since(a).as_secs_f64() * 1000.0)
        }

        let current_frame_index = self.current_frame_index as usize;
        let current_frame_future = &mut self.swapchain_info.in_flight_frames[0];

        current_frame_future.cleanup_finished();


        if is_suboptimal {
            self.request_recreate_swapchain();
        }

        self.swapchain_info.current_image_idx = image_index;
        self.swapchain_info.acquire_future = Some(acquire_future);

        let cmd_buf = match AutoCommandBufferBuilder::primary(allocator, queue_family_index, CommandBufferUsage::OneTimeSubmit) {
            Ok(r) => r,
            Err(err) => return BeginFrameResult::Err(log_error_and_throw!(anyhow!(err), "Failed to allocate command buffer for begin_frame"))
        };

        BeginFrameResult::Begin(cmd_buf)
    }

    pub fn present_frame(&mut self, cmd_buf: PrimaryCommandBuffer) -> Result<bool> {
        let cmd_buf = cmd_buf.build()?;

        let device = self.device.clone();
        let swapchain = self.get_swapchain()?.clone();
        let image_index = self.swapchain_info.current_image_idx;
        let acquire_future = self.swapchain_info.acquire_future.take().unwrap();

        let current_frame_index = self.current_frame_index as usize;

        let queue = self.queues.get(QueueId::GraphicsMain.name())
            .ok_or_else(|| anyhow!("Failed to get the GRAPHICS queue \"{}\"", QueueId::GraphicsMain.name()))?;

        let present_info = SwapchainPresentInfo::swapchain_image_index(swapchain, image_index);


        let a = Instant::now();
        let future = acquire_future
            .then_execute(queue.clone(), cmd_buf)?
            .then_swapchain_present(queue.clone(), present_info)
            .then_signal_fence_and_flush();

        let future = match future {
            Ok(future) => {
                Box::new(future) as Box<dyn GpuFuture>
            }
            Err(Validated::Error(VulkanError::OutOfDate)) => {
                self.request_recreate_swapchain();
                self.sync_now()
            }
            Err(err) => {
                error!("Failed to flush future: {:?}", err);
                self.sync_now()
            }
        };

        self.swapchain_info.in_flight_frames[0] = future;

        let b = Instant::now();

        if b.duration_since(a).as_secs_f64() > 0.001 {
            debug!("signal_fence_and_flush blocked for {:.4} msec", b.duration_since(a).as_secs_f64() * 1000.0)
        }

        self.swapchain_info.prev_image_idx = self.swapchain_info.current_image_idx;
        self.swapchain_info.current_image_idx = image_index;
        self.current_frame_index = (self.current_frame_index + 1) % self.max_concurrent_frames;
        Ok(true)
    }

    pub fn flush_rendering(&mut self) -> Result<()>{
        for frame_future in self.swapchain_info.in_flight_frames.iter_mut() {
            frame_future.cleanup_finished();

            let mut prev_frame_end = Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>;
            mem::swap(&mut prev_frame_end, frame_future);
            prev_frame_end.cleanup_finished();
        }

        unsafe { self.device.wait_idle() }?;

        Ok(())
    }

    pub fn sync_now(&self) -> Box<dyn GpuFuture> {
        Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>
    }

    fn get_swapchain(&self) -> Result<&Arc<Swapchain>> {
        let swapchain = self.swapchain_info.swapchain.as_ref()
            .ok_or_else(|| anyhow!("Failed to get swapchain"))?;
        Ok(swapchain)
    }

    pub fn get_render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.as_ref().unwrap().clone()
    }

    pub fn get_queue_details(&self) -> &QueueDetails {
        &self.queue_details
    }

    // pub fn get_command_pool(&self, queue_family_index: u32) -> Option<&CommandPool> {
    //     self.command_pools.get(&queue_family_index)
    // }
    // 
    // pub fn get_command_pool_mut(&mut self, queue_family_index: u32) -> Option<&mut CommandPool> {
    //     self.command_pools.get_mut(&queue_family_index)
    // }
    // 
    // pub fn get_graphics_command_pool(&self) -> &CommandPool {
    //     // We expect to always have a graphics command pool, these unwraps are fine.
    //     self.get_command_pool(self.queue_details.graphics_queue_family_index.unwrap()).unwrap()
    // }
    // 
    // pub fn get_graphics_command_pool_mut(&mut self) -> &mut CommandPool {
    //     self.get_command_pool_mut(self.queue_details.graphics_queue_family_index.unwrap()).unwrap()
    // }

    pub fn request_recreate_swapchain(&mut self) {
        if self.update_swapchain_request.is_none() {
            self.update_swapchain_request = Some(UpdateSwapchainRequest{
                image_extent: self.swapchain_info.image_extent,
            });
        }
    }

    pub fn set_resolution(&mut self, width: u32, height: u32) -> Result<bool> {
        if width == self.swapchain_info.image_extent[0] && height == self.swapchain_info.image_extent[1] {
            return Ok(false); // width & height is unchanged
        }

        self.request_recreate_swapchain();
        let a = self.update_swapchain_request.as_mut().unwrap();
        a.image_extent[0] = width;
        a.image_extent[1] = height;

        Ok(true)
    }

    pub fn get_resolution(&self) -> [u32; 2] {
        self.swapchain_info.image_extent
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
    
    pub fn get_current_framebuffer(&self) -> Arc<Framebuffer> {
        self.swapchain_info.framebuffers.get(self.swapchain_info.current_image_idx as usize).unwrap().clone()
    }

    pub fn get_max_concurrent_frames(&self) -> u32 {
        self.max_concurrent_frames
    }

    pub fn get_current_frame_index(&self) -> u32 {
        self.current_frame_index
    }

    // pub fn get_current_graphics_cmd_buffer(&self) -> &Arc<CommandBuffer> {
    //     let a = self.swapchain_info.command_buffers.get(self.current_frame_index as usize)
    //         .ok_or_else(|| anyhow!("Failed to get command buffer for frame {}", self.swapchain_info.current_frame_idx));
    //     
    //     a.unwrap()
    // }
}




#[derive(Default, Debug)]
pub struct QueueDetails {
    pub graphics_queue_family_index: Option<u32>,
    pub compute_queue_family_index: Option<u32>,
    pub transfer_queue_family_index: Option<u32>,
    pub sparse_binding_queue_family_index: Option<u32>,
    pub protected_queue_family_index: Option<u32>,
    pub present_queue_family_index: Option<u32>,
}

impl QueueDetails {
    pub fn indices(&self) -> [Option<u32>; 6] {
        [
            self.graphics_queue_family_index,
            self.compute_queue_family_index,
            self.transfer_queue_family_index,
            self.sparse_binding_queue_family_index,
            self.protected_queue_family_index,
            self.present_queue_family_index,
        ]
    }

    pub fn unique_indices(&self) -> HashSet<u32> {
        self.indices().into_iter().flatten().collect()
    }
}
