use crate::metal::CommandBuffer;
use crate::metal::MetalGPU;
use block2::RcBlock;
use dispatch2::DispatchObject;
use dispatch2::DispatchSemaphore;
use dispatch2::DispatchTime;
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
    let mut gpu = MetalGPU::new_metal_gpu().unwrap();
    let first = String::from("first");
    gpu.new_command_queue(&first, Some(true)).unwrap();
    if gpu.metal4_supported {
        gpu.new_command_allocator(&first).unwrap();
    }

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
    let command_queue = gpu.get_command_queue(&first).unwrap();
    let args = &[
        (input_buffer.as_ref(), 0, 0),
        (output_buffer.as_ref(), 0, 1),
        (scale_buffer.as_ref(), 0, 2),
    ];
    match command_queue {
        metal::CommandQueue::Metal(cq) => {
            let command_buffer = gpu.new_command_buffer(cq).unwrap();

            command_buffer
                .fill_with_arguments(
                    &scale_tensor,
                    args,
                    grid_size,
                    threadgroup_size,
                    Some(MTLDispatchType::Serial),
                    None,
                )
                .unwrap();

            unsafe {
                CommandBuffer::commit(&[&command_buffer], None, None).unwrap();
            }

            let command_buffer = command_buffer.as_metal().unwrap();
            command_buffer.waitUntilScheduled();
            println!("Scheduled.");
            command_buffer.waitUntilCompleted();
            println!("Completed.");
        }
        metal::CommandQueue::Metal4(cq) => {
            let allocator = gpu.get_command_allocator(&first).unwrap();
            let command_buffer = gpu.new_command_buffer_metal_4(allocator).unwrap();
            command_buffer
                .fill_with_arguments(
                    &scale_tensor,
                    args,
                    grid_size,
                    threadgroup_size,
                    None,
                    Some(&gpu),
                )
                .unwrap();
            command_buffer.end_command_buffer_metal_4();

            let sem = DispatchSemaphore::new(0);
            let sem_clone = sem.retain();
            let callback =
                RcBlock::new(move |x: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                    let feedback = unsafe { x.as_ref() };
                    println!(
                        "Committed: start {:?}, end {:?}, diff {:?}",
                        feedback.GPUStartTime(),
                        feedback.GPUEndTime(),
                        feedback.GPUEndTime() - feedback.GPUStartTime(),
                    );
                    sem_clone.signal();
                });

            unsafe {
                CommandBuffer::commit(
                    &[&command_buffer],
                    Some(cq),
                    Some(RcBlock::as_ptr(&callback)),
                )
                .unwrap();
            }

            // Block until semaphore is done.
            sem.try_acquire(DispatchTime::FOREVER).unwrap();
        }
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
    let file = File::open("./models/TinyMistral-248M-v3/model.safetensors").unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    let names = tensors.names();
    println!("{:?}", names);
    let tensor = tensors
        .tensor("model.layers.2.self_attn.k_proj.weight")
        .unwrap();
    println!("{:?}", tensor.shape())
}
