use crate::metal::MetalGPU;
use memmap2::MmapOptions;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::fs::File;
use std::ptr::NonNull;

pub use safetensors::SafeTensors;
pub use safetensors::serialize;

mod embeddings;
mod metal;
mod model;

fn main() {
    let gpu = MetalGPU::new_metal_gpu().unwrap();
    gpu.new_command_queue(String::from("first")).unwrap();

    println!("device: {:?}", gpu.device.name());
    println!("metal4 supported: {:?}", gpu.metal4_supported);

    // Load the compiled metallib
    let scale_tensor = gpu
        .load_kernel_file(
            String::from("./src/kernels/kernels.metallib"),
            String::from("scale_tensor"),
        )
        .unwrap();

    // --- Set up test data ---
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let scale: f32 = 2.5;
    let count = input.len();
    let buffer_size = count * std::mem::size_of::<f32>();

    // Declare buffers
    let input_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;
    let output_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;
    let scale_buffer: objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>>;

    unsafe {
        input_buffer = gpu
            .device
            .newBufferWithBytes_length_options(
                NonNull::new(input.as_ptr() as *mut _).unwrap(),
                buffer_size,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("failed to create input buffer");
    };

    output_buffer = gpu
        .device
        .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
        .expect("failed to create output buffer");

    unsafe {
        // Create scale buffer
        scale_buffer = gpu
            .device
            .newBufferWithBytes_length_options(
                NonNull::new(&scale as *const f32 as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("failed to create scale buffer");
    }

    // Encode and dispatch
    let command_buffer = command_queue
        .commandBuffer()
        .expect("failed to create command buffer");

    let encoder = command_buffer
        .computeCommandEncoder()
        .expect("failed to create compute command encoder");

    encoder.setComputePipelineState(&pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&scale_buffer), 0, 2);
    }

    let grid_size = MTLSize {
        width: count as usize,
        height: 1,
        depth: 1,
    };

    let threadgroup_size = MTLSize {
        width: pipeline.maxTotalThreadsPerThreadgroup().min(count as usize),
        height: 1,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

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
