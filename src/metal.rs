use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTL4CommandQueue, MTLBuffer, MTLCommandQueue, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary, MTLResourceOptions,
    MTLTensorDescriptor,
};
use std::{collections::HashMap, ffi::c_void, fs::File, io::Read, ptr::NonNull};

use crate::metal;

pub fn setup_device()
-> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>, String> {
    MTLCreateSystemDefaultDevice().ok_or("unable to create device".to_string())
}

#[derive(Debug)]
pub enum MetalGPUError {
    IoError(std::io::Error),
    NSError(Retained<NSError>),
    DeviceError(String),
    FunctionError(String),
    KeyError(String),
    CommandQueueCreationError(String),
    BufferCreationError(String),
}

impl std::fmt::Display for MetalGPUError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalGPUError::IoError(error) => error.fmt(f),
            MetalGPUError::NSError(retained) => retained.fmt(f),
            MetalGPUError::DeviceError(err_str) => write!(f, "DeviceError: {}", err_str),
            MetalGPUError::FunctionError(err_str) => write!(f, "FunctionError: {}", err_str),
            MetalGPUError::BufferCreationError(err_str) => {
                write!(f, "BufferCreationError: {}", err_str)
            }
            MetalGPUError::KeyError(err_str) => {
                write!(f, "KeyError: {}", err_str)
            }
            MetalGPUError::CommandQueueCreationError(err_str) => {
                write!(f, "CommandQueueCreationError: {}", err_str)
            }
        }
    }
}

impl From<std::io::Error> for MetalGPUError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError(error)
    }
}
impl From<Retained<NSError>> for MetalGPUError {
    fn from(error: Retained<NSError>) -> Self {
        Self::NSError(error)
    }
}

// Capturing hardware variance. Queues enable instruction pipelining.
pub enum CommandQueue {
    Metal(Retained<ProtocolObject<dyn MTLCommandQueue>>),
    Metal4(Retained<ProtocolObject<dyn MTL4CommandQueue>>),
}

pub struct MetalGPU {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub metal4_supported: bool,

    // Named queues allow for smart instruction routing.
    queues: HashMap<String, CommandQueue>,
}

impl MetalGPU {
    // new_metal_gpu creates a MetalGPU object.
    pub fn new_metal_gpu() -> Result<MetalGPU, MetalGPUError> {
        let device = metal::setup_device().map_err(|e| MetalGPUError::DeviceError(e))?;
        let metal4_supported = device.supportsFamily(MTLGPUFamily::Metal4);
        Ok(MetalGPU {
            device,
            metal4_supported,
            queues: HashMap::new(),
        })
    }

    // load_kernel_file reads and loads a metallib file, generating a MTLComputePipelinState or
    // error.
    pub fn load_kernel_file(
        &self,
        filename: &String,
        libname: &String,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalGPUError> {
        let mut buffer = Vec::new();
        let mut file = File::open(filename)?;
        file.read_to_end(&mut buffer).map_err(MetalGPUError::from)?;
        let library = self
            .device
            .newLibraryWithData_error(&DispatchData::from_bytes(&buffer))?;
        let f = library
            .newFunctionWithName(&NSString::from_str(libname.as_str()))
            .ok_or_else(|| {
                MetalGPUError::FunctionError(format!(
                    "Failed to find Metal function '{}' in the compiled library",
                    libname
                ))
            })?;
        let function = self.device.newComputePipelineStateWithFunction_error(&f)?;
        Ok(function)
    }

    // new_command_queue instantiates a command queue, which is Metal 4 aware.
    // Coerce forcing enabling or disabling metal4 support via metal4 argument.
    pub fn new_command_queue(
        &mut self,
        name: &String,
        metal4: Option<bool>,
    ) -> Result<(), MetalGPUError> {
        // Check whether the queue already exists, if so error.
        if self.queues.contains_key(name) {
            return Err(MetalGPUError::KeyError(format!("Key {}", name)));
        }

        // If metal4 is not set, default to device capabilities, preferring metal4.
        let metal4_supported = metal4.map_or(self.metal4_supported, |x| x);

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
        let queue = queue
            .ok_or_else(|| MetalGPUError::CommandQueueCreationError(format!("Key {}", name)))?;
        self.queues.insert(name.clone(), queue);
        Ok(())
    }

    pub fn get_command_queue(&self, name: &String) -> Result<&CommandQueue, MetalGPUError> {
        self.queues.get(name).ok_or(MetalGPUError::KeyError(format!(
            "Key {} does not exist",
            name
        )))
    }

    pub unsafe fn new_buffer_from_bytes(
        &self,
        pointer: NonNull<c_void>,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalGPUError> {
        let buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(pointer, length, options)
        };
        buf.ok_or(MetalGPUError::BufferCreationError(format!(
            "Unable to create buffer from {:?} of length {:?} with options {:?}",
            pointer.addr().get(),
            length,
            options
        )))
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalGPUError> {
        self.device
            .newBufferWithLength_options(length, options)
            .ok_or(MetalGPUError::BufferCreationError(format!(
                "Unable to create buffer of length {:?} with options {:?}",
                length, options
            )))
    }
}

pub fn new_tensor_descriptor() -> Retained<MTLTensorDescriptor> {
    // TODO @cyrusknopf pass args
    return MTLTensorDescriptor::new();
}
