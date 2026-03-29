# Tensors

We will look  at tensors from the perspective of the host (Rust), and of the device (Metal kernels).

## Tensors on the host

Apple Metal provides an abstraction of a tensor over a buffer of bytes.

To create a tensor, we use the function `MTLDevice::newTensorWithDescriptor_error`(). This function takes a `MTLTensorDescriptor`.

This descriptor specifies properties of the tensor to create, such as data type (e.g. float16, int, etc), dimensions, storage mode (configures memory location, access permissions), and CPU cache mode (tbd).

We implement the function `new_tensor_descriptor` to instantiate descriptors.

`MTLDevice::newTensorWithDescriptor_error` returns some wrapping of a `MTLTensor`, from which we can access the raw buffer, dimensions, datatypes, etc.

We can pass these buffers directly to metal shaders.

## Tensors on the device

Below is a function from [a repo with some examples of using Metal tensors](https://github.com/liuliu/example_matmul_metal4/blob/main/Sources/matmul/shader.metal):
~~~cpp
#include <metal_stdlib>
#include <metal_tensor>

...

kernel void matmul_static_slice_static_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
	// Use static extents. Note that these shapes are template parameters, it is fixed at compile-time.
    auto A = tensor<device half,  extents<int32_t, 256, 128>, tensor_inline>(A_buf, extents<int32_t, 256, 128>());
    auto B = tensor<device half,  extents<int32_t, 64, 256>, tensor_inline>(B_buf, extents<int32_t, 64, 256>());
    auto C = tensor<device half,  extents<int32_t, 128, 64>, tensor_inline>(C_buf, extents<int32_t, 128, 64>());
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    for (int k = 0; k < 256; k += 16) {
        // Create appropriate slice for this thread group to work on.
        auto mA = A.slice<16, 64>(k, tgid.y * 64);
        auto mB = B.slice<32, 16>(tgid.x * 32, k);
        auto mC = C.slice<32, 64>(tgid.x * 32, tgid.y * 64);

        // execute the operation. Assumes C is is initialized to zero.
        matmulOp.run(mA, mB, mC);
    }
}
~~~

We can see the `<metal_tensor>` header provides an abstraction over interpreting a raw buffer as a tensor, and the matrix multiplication operation on those tensors. The examples uses static dimensions, but if we were to write a dynamic version, we can sync the `MTLTensor::dimensions` struct with the dimensions passed to the kernel.


## Example

A crude unsafe example of creating a tensor on the host:

~~~rust
let mut gpu = MetalGPU::new_metal_gpu().unwrap();
let desc: Retained<MTLTensorDescriptor> = new_tensor_descriptor();
let device = gpu.device;
let maybe_tensor = device.newTensorWithDescriptor_error(desc.deref());
let _tensor = maybe_tensor.unwrap();
~~~
