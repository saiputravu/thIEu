use crate::metal::MetalGPU;
use memmap2::MmapOptions;
use objc2::rc::Retained;
use objc2::runtime::MessageReceiver;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSObjectProtocol;
use objc2_metal::*;
use std::fs::File;
use std::ptr::NonNull;

pub use safetensors::SafeTensors;
pub use safetensors::serialize;

mod embeddings;
mod metal;
mod model;

fn main() {
    let mut gpu = MetalGPU::new_metal_gpu().unwrap();
    let queue_name = String::from("first");
    gpu.new_command_queue(&queue_name, Some(false)).unwrap();

    println!("device: {:?}", gpu.device.name());
    println!("metal4 supported: {:?}", gpu.metal4_supported);

    // Load the compiled metallib
    let scale_tensor = gpu
        .load_kernel_file(
            &String::from("./src/kernels/kernels.metallib"),
            &String::from("scale_tensor"),
        )
        .unwrap();

    // --- Set up test data ---
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scale: f32 = 2.5;
    let count = input.len();
    let buffer_size = count * std::mem::size_of::<f32>();
    let count_size = std::mem::size_of::<f32>();

    // Declare buffers
    let input_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;
    let output_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;
    let scale_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;

    input_buffer = unsafe {
        let ptr = NonNull::new(input.as_ptr() as *mut _).expect("failed to cast to pointer");
        gpu.new_buffer_from_bytes(ptr, buffer_size, MTLResourceOptions::StorageModeShared)
    }
    .unwrap();

    scale_buffer = unsafe {
        let ptr = NonNull::new(&scale as *const f32 as *mut _).expect("failed to cast to pointer");
        gpu.new_buffer_from_bytes(ptr, count_size, MTLResourceOptions::StorageModeShared)
    }
    .unwrap();

    output_buffer = gpu
        .new_buffer(buffer_size, MTLResourceOptions::StorageModeShared)
        .unwrap();

    let grid_size = MTLSize {
        width: count as usize,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: scale_tensor
            .maxTotalThreadsPerThreadgroup()
            .min(count as usize),
        height: 1,
        depth: 1,
    };

    // Encode and dispatch
    let command_queue = gpu.get_command_queue(&queue_name).unwrap();
    match command_queue {
        metal::CommandQueue::Metal(cq) => {
            let command_buffer = cq.commandBuffer().unwrap();

            // Setup the command buffer.
            let command_encoder = command_buffer.computeCommandEncoder().unwrap();
            command_encoder.setComputePipelineState(&scale_tensor);
            unsafe {
                command_encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
                command_encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
                command_encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 2);
            }
            command_encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
            command_encoder.endEncoding();

            // Commit and wait till completion.
            command_buffer.commit();
            command_buffer.waitUntilScheduled();
            println!("Scheduled.");
            command_buffer.waitUntilCompleted();
            println!("Completed.");
        }
        metal::CommandQueue::Metal4(cq) => {
            panic!("Unimplemented")
        } // metal::CommandQueue::Metal4(cq) => cq     };
    };

    // Read back results
    let output_ptr = output_buffer.contents().as_ptr() as *const f32;

    let output: Vec<f32>;
    unsafe {
        output = std::slice::from_raw_parts(output_ptr, count).to_vec();
    }

    println!("input:  {:?}", input);
    println!("scale:  {}", scale);
    println!("output: {:?}", output);

    // Read safetensors.
    let file = File::open("./models/LFM2.5-1.2B-Thinking/model.safetensors").unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    let names = tensors.names();
    println!("{:?}", names);
    let tensor = tensors
        .tensor("model.layers.2.self_attn.k_proj.weight")
        .unwrap();
    println!("{:?}", tensor.shape())
}
