use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedbackHandler, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLDispatchType, MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLSize, MTLTensorDescriptor,
};
use std::{collections::HashMap, ffi::c_void, fs::File, io::Read, ptr::NonNull};

/// Errors produced by Metal GPU operations.
#[derive(Debug)]
pub enum MetalError {
    IoError(std::io::Error),
    NSError(Retained<NSError>),
    DeviceError(String),
    FunctionError(String),
    KeyError(String),
    CommandQueueCreationError(String),
    BufferCreationError(String),
    CommandBufferCreationError(&'static str),
    CommandAllocatorCreationError(String),
    CommandBufferError(String),
    Metal4NotSupportedError,
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::IoError(error) => error.fmt(f),
            MetalError::NSError(retained) => retained.fmt(f),
            MetalError::DeviceError(err_str) => write!(f, "DeviceError: {}", err_str),
            MetalError::FunctionError(err_str) => write!(f, "FunctionError: {}", err_str),
            MetalError::BufferCreationError(err_str) => {
                write!(f, "BufferCreationError: {}", err_str)
            }
            MetalError::CommandBufferCreationError(err_str) => {
                write!(f, "CommandBufferCreationError: {}", err_str)
            }
            MetalError::CommandAllocatorCreationError(err_str) => {
                write!(f, "CommandAllocatorCreationError: {}", err_str)
            }
            MetalError::KeyError(err_str) => {
                write!(f, "KeyError: {}", err_str)
            }
            MetalError::CommandQueueCreationError(err_str) => {
                write!(f, "CommandQueueCreationError: {}", err_str)
            }
            MetalError::CommandBufferError(err_str) => {
                write!(f, "CommandBufferError: {}", err_str)
            }
            MetalError::Metal4NotSupportedError => {
                write!(f, "Metal4NotSupportedError")
            }
        }
    }
}

impl std::error::Error for MetalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MetalError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for MetalError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}
impl From<Retained<NSError>> for MetalError {
    fn from(error: Retained<NSError>) -> Self {
        Self::NSError(error)
    }
}

/// A command queue abstraction over Metal and Metal4 hardware variants.
pub enum CommandQueue {
    Metal(Retained<ProtocolObject<dyn MTLCommandQueue>>),
    Metal4(Retained<ProtocolObject<dyn MTL4CommandQueue>>),
}

impl CommandQueue {
    pub fn as_metal(&self) -> Option<&ProtocolObject<dyn MTLCommandQueue>> {
        match self {
            CommandQueue::Metal(cq) => Some(cq),
            _ => None,
        }
    }

    pub fn as_metal_4(&self) -> Option<&ProtocolObject<dyn MTL4CommandQueue>> {
        match self {
            CommandQueue::Metal4(cq) => Some(cq),
            _ => None,
        }
    }
}

/// A command buffer abstraction over Metal and Metal4 hardware variants.
pub enum CommandBuffer {
    Metal(Retained<ProtocolObject<dyn MTLCommandBuffer>>),
    Metal4(Retained<ProtocolObject<dyn MTL4CommandBuffer>>),
}

impl CommandBuffer {
    /// Encodes a compute pass into this command buffer.
    ///
    /// Binds `arguments` (buffer, offset, index) to a compute encoder, sets the pipeline state,
    /// dispatches the threadgroups, and ends encoding. For Metal, `dispatch_type` is required.
    /// For Metal4, `gpu` is required (to create an argument table).
    pub fn fill_with_arguments(
        &self,
        computation: &ProtocolObject<dyn MTLComputePipelineState>,
        arguments: &[(&ProtocolObject<dyn MTLBuffer>, usize, usize)],
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
        dispatch_type: Option<MTLDispatchType>, // Only matters in Metal
        gpu: Option<&MetalGPU>,                 // Required for the Metal4 path
    ) -> Result<(), MetalError> {
        match self {
            CommandBuffer::Metal(cb) => {
                let dispatch_type = dispatch_type.ok_or(MetalError::DeviceError(String::from(
                    "Dispatch type needed for metal",
                )))?;
                let encoder = cb
                    .computeCommandEncoderWithDispatchType(dispatch_type)
                    .ok_or(MetalError::CommandBufferError(String::from(
                        "Cannot create encoder",
                    )))?;

                // Setup the arguments in the command encoder.
                for (buffer, offset, index) in arguments {
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(buffer), *offset, *index);
                    }
                }

                encoder.setComputePipelineState(computation);
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    threadgroups_per_grid,
                    threads_per_threadgroup,
                );
                encoder.endEncoding();
                Ok(())
            }
            CommandBuffer::Metal4(cb) => {
                let gpu = gpu.ok_or(MetalError::DeviceError(String::from(
                    "fill_with_arguments requires device",
                )))?;

                let argument_table = gpu.new_argument_table(arguments.len())?;
                for (buffer, offset, index) in arguments {
                    unsafe {
                        argument_table
                            .setAddress_atIndex(buffer.gpuAddress() + (*offset as u64), *index);
                    }
                }

                let encoder = cb
                    .computeCommandEncoder()
                    .ok_or(MetalError::CommandBufferError(String::from(
                        "Cannot create encoder",
                    )))?;

                encoder.setComputePipelineState(computation);
                encoder.setArgumentTable(Some(&argument_table));
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    threadgroups_per_grid,
                    threads_per_threadgroup,
                );
                encoder.endEncoding();
                Ok(())
            }
        }
    }

    /// Commits one or more command buffers for execution.
    ///
    /// Metal command buffers are committed individually. Metal4 command buffers are
    /// batched and committed together through the provided `command_queue`.
    ///
    /// # Safety
    /// Metal4 command buffers must have had [`end_command_buffer_metal_4`](Self::end_command_buffer_metal_4)
    /// called before commit.
    pub unsafe fn commit(
        command_buffers: &[&CommandBuffer],
        command_queue: Option<&ProtocolObject<dyn MTL4CommandQueue>>,
        callback: Option<MTL4CommitFeedbackHandler>,
    ) -> Result<(), MetalError> {
        if command_buffers.is_empty() {
            return Err(MetalError::CommandBufferError(String::from(
                "empty command buffers",
            )));
        }

        let mut command_buffers_metal_4: Vec<NonNull<ProtocolObject<dyn MTL4CommandBuffer>>> =
            Vec::new();
        for command_buffer in command_buffers {
            match command_buffer {
                CommandBuffer::Metal(cb) => cb.commit(),
                CommandBuffer::Metal4(cb) => {
                    command_buffers_metal_4.push(NonNull::from(cb.as_ref()))
                }
            };
        }

        // If there are MTL4CommandBuffers to commit, commit them.
        if !command_buffers_metal_4.is_empty() {
            let options = MTL4CommitOptions::new();
            callback.map(|f| unsafe { options.addFeedbackHandler(f) });

            let command_buffer_metal_4_ptr = NonNull::new(command_buffers_metal_4.as_mut_ptr())
                .ok_or(MetalError::CommandBufferCreationError(
                    "Cannot create nonnull to command buffers",
                ))?;
            command_queue.map(|cq| unsafe {
                cq.commit_count_options(
                    command_buffer_metal_4_ptr,
                    command_buffers_metal_4.len(),
                    &options,
                )
            });
        }
        Ok(())
    }

    /// Finalizes a Metal4 command buffer before commit. No-op for Metal command buffers.
    pub fn end_command_buffer_metal_4(&self) {
        match self {
            CommandBuffer::Metal4(cb) => cb.endCommandBuffer(),
            CommandBuffer::Metal(_) => (),
        }
    }

    pub fn as_metal(&self) -> Option<&ProtocolObject<dyn MTLCommandBuffer>> {
        match self {
            CommandBuffer::Metal(command_buffer) => Some(command_buffer),
            _ => None,
        }
    }

    pub fn as_metal_4(&self) -> Option<&ProtocolObject<dyn MTL4CommandBuffer>> {
        match self {
            CommandBuffer::Metal4(command_buffer) => Some(command_buffer),
            _ => None,
        }
    }
}

/// The primary GPU device handle, managing command queues, allocators, and buffers.
pub struct MetalGPU {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub metal4_supported: bool,

    /// Named queues for instruction routing.
    queues: HashMap<String, CommandQueue>,
    allocators: HashMap<String, Retained<ProtocolObject<dyn MTL4CommandAllocator>>>,
}

impl MetalGPU {
    /// Creates a new `MetalGPU` using the system default Metal device.
    pub fn new() -> Result<MetalGPU, MetalError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MetalError::DeviceError("unable to create device".to_string()))?;
        let metal4_supported = device.supportsFamily(MTLGPUFamily::Metal4);
        Ok(MetalGPU {
            device,
            metal4_supported,
            queues: HashMap::new(),
            allocators: HashMap::new(),
        })
    }

    /// Loads a compiled `.metallib` file and extracts a compute pipeline for the named function.
    pub fn load_kernel_file(
        &self,
        filename: &str,
        libname: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        let mut buffer = Vec::new();
        let mut file = File::open(filename)?;
        file.read_to_end(&mut buffer).map_err(MetalError::from)?;
        let library = self
            .device
            .newLibraryWithData_error(&DispatchData::from_bytes(&buffer))?;
        let f = library
            .newFunctionWithName(&NSString::from_str(libname))
            .ok_or_else(|| {
                MetalError::FunctionError(format!(
                    "Failed to find Metal function '{}' in the compiled library",
                    libname
                ))
            })?;
        let function = self.device.newComputePipelineStateWithFunction_error(&f)?;
        Ok(function)
    }

    /// Creates and stores a named command queue, selecting Metal or Metal4 based on
    /// device capabilities. Override with `metal4` to force a specific variant.
    pub fn new_command_queue(
        &mut self,
        name: &str,
        metal4: Option<bool>,
    ) -> Result<(), MetalError> {
        // Check whether the queue already exists, if so error.
        if self.queues.contains_key(name) {
            return Err(MetalError::CommandQueueCreationError(format!(
                "Key already exists {}",
                name
            )));
        }

        // If metal4 is not set, default to device capabilities, preferring metal4.
        let metal4_supported = metal4.unwrap_or(self.metal4_supported);

        // Create a command queue based on supported hardware features.
        let queue = if metal4_supported {
            self.device
                .newMTL4CommandQueue()
                .map(|q| CommandQueue::Metal4(q))
        } else {
            self.device
                .newCommandQueue()
                .map(|q| CommandQueue::Metal(q))
        };
        let queue = queue.ok_or_else(|| {
            MetalError::CommandQueueCreationError(format!("Failed insert Key {}", name))
        })?;
        self.queues.insert(name.to_owned(), queue);
        Ok(())
    }

    /// Returns a reference to the named command queue, or `KeyError` if not found.
    pub fn get_command_queue(&self, name: &str) -> Result<&CommandQueue, MetalError> {
        self.queues
            .get(name)
            .ok_or(MetalError::KeyError(format!("Key {} does not exist", name)))
    }

    /// Creates a GPU buffer by copying from the given pointer.
    ///
    /// # Safety
    /// `pointer` must be valid for `length` bytes.
    pub unsafe fn new_buffer_from_bytes(
        &self,
        pointer: NonNull<c_void>,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        let buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(pointer, length, options)
        };
        buf.ok_or(MetalError::BufferCreationError(format!(
            "Unable to create buffer from {:?} of length {:?} with options {:?}",
            pointer.addr().get(),
            length,
            options
        )))
    }

    /// Allocates a new GPU buffer of the given length and resource options.
    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        self.device
            .newBufferWithLength_options(length, options)
            .ok_or(MetalError::BufferCreationError(format!(
                "Unable to create buffer of length {:?} with options {:?}",
                length, options
            )))
    }

    /// Creates a Metal command buffer from the given command queue.
    ///
    /// For Metal4 command buffers, use [`new_command_buffer_metal_4`](Self::new_command_buffer_metal_4).
    pub fn new_command_buffer(
        &self,
        command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> Result<CommandBuffer, MetalError> {
        command_queue
            .commandBuffer()
            .map(CommandBuffer::Metal)
            .ok_or(MetalError::CommandBufferCreationError(
                "Unable to create command buffer",
            ))
    }

    /// Creates a Metal4 command buffer using the given allocator.
    ///
    /// Returns `Metal4NotSupportedError` if the device does not support Metal4.
    /// For standard Metal command buffers, use [`new_command_buffer`](Self::new_command_buffer).
    pub fn new_command_buffer_metal_4(
        &self,
        allocator_ref: &Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    ) -> Result<CommandBuffer, MetalError> {
        // Check metal supported
        if !self.metal4_supported {
            return Err(MetalError::Metal4NotSupportedError);
        }
        let command_buffer =
            self.device
                .newCommandBuffer()
                .ok_or(MetalError::CommandBufferCreationError(
                    "Unable to create command buffer",
                ))?;
        command_buffer.beginCommandBufferWithAllocator(allocator_ref);
        Ok(CommandBuffer::Metal4(command_buffer))
    }

    /// Creates and stores a named Metal4 command allocator.
    ///
    /// Metal4 only. Returns an error if the name is already taken or Metal4 is unsupported.
    pub fn new_command_allocator(&mut self, name: &str) -> Result<(), MetalError> {
        // Check metal supported
        if !self.metal4_supported {
            return Err(MetalError::Metal4NotSupportedError);
        }
        if self.allocators.contains_key(name) {
            return Err(MetalError::CommandAllocatorCreationError(format!(
                "Key {} already exists",
                name
            )));
        }

        // Create and track allocator.
        let allocator =
            self.device
                .newCommandAllocator()
                .ok_or(MetalError::CommandAllocatorCreationError(format!(
                    "Failed to create allocator {}",
                    name
                )))?;
        self.allocators.insert(name.to_owned(), allocator);
        Ok(())
    }

    /// Returns a reference to the named Metal4 command allocator, or `KeyError` if not found.
    pub fn get_command_allocator(
        &self,
        name: &str,
    ) -> Result<&Retained<ProtocolObject<dyn MTL4CommandAllocator>>, MetalError> {
        self.allocators
            .get(name)
            .ok_or(MetalError::KeyError(format! {"Key error {}", name}))
    }

    /// Creates a Metal4 argument table with the given buffer bind count.
    ///
    /// Metal4 only. Used internally by [`CommandBuffer::fill_with_arguments`] for the Metal4 path.
    pub fn new_argument_table(
        &self,
        max_buffer_bind_count: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTL4ArgumentTable>>, MetalError> {
        if !self.metal4_supported {
            return Err(MetalError::Metal4NotSupportedError);
        }

        let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
        argument_table_descriptor.setMaxBufferBindCount(max_buffer_bind_count);
        let argument_table = self
            .device
            .newArgumentTableWithDescriptor_error(&argument_table_descriptor)?;
        Ok(argument_table)
    }

    /// Creates a new tensor descriptor.
    // TODO(@cyrusknopf): Accept shape/datatype arguments.
    pub fn new_tensor_descriptor(&self) -> Retained<MTLTensorDescriptor> {
        MTLTensorDescriptor::new()
    }
}
