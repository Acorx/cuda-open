"""
CUDA Open Compiler - Multi-Target Backends

Définit les spécificités syntaxiques pour générer du code natif sur :
1. CUDA (NVIDIA)
2. HIP (AMD ROCm)
3. SYCL (Intel / Standard)
4. C++ Standard (CPU Fallback)
"""

class BackendSpec:
    def __init__(self, name, kernel_macro, grid_macro, block_macro, sync_func, headers):
        self.name = name
        self.kernel_macro = kernel_macro # Pour marquer une fonction kernel
        self.grid_macro = grid_macro     # Pour l'ID global (ex: blockIdx.x * blockDim.x + threadIdx.x)
        self.block_macro = block_macro   # Pour l'ID local
        self.sync_func = sync_func       # Synchronisation thread (ex: __syncthreads())
        self.headers = headers           # Includes nécessaires

# Définitions des Backends

CUDA_BACKEND = BackendSpec(
    name="CUDA (NVIDIA)",
    kernel_macro="__global__ void",
    grid_macro="(blockIdx.x * blockDim.x + threadIdx.x)",
    block_macro="threadIdx.x",
    sync_func="__syncthreads()",
    headers=["#include <cuda_runtime.h>", "#include <stdio.h>"]
)

HIP_BACKEND = BackendSpec(
    name="HIP (AMD ROCm)",
    kernel_macro="__global__ void",
    grid_macro="(hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x)",
    block_macro="hipThreadIdx_x",
    sync_func="__syncthreads()",
    headers=["#include <hip/hip_runtime.h>", "#include <stdio.h>"]
)

SYCL_BACKEND = BackendSpec(
    name="SYCL (Intel/Standard)",
    kernel_macro="void", # En SYCL c'est souvent un lambda ou une classe, ici on simule une fonction C-style pour l'interop
    grid_macro="sycl::item<1> item = sycl::ext::oneapi::experimental::this_nd_item<1>(); item.get_global_id(0)",
    block_macro="sycl::item<1> item = sycl::ext::oneapi::experimental::this_nd_item<1>(); item.get_local_id(0)",
    sync_func="item.barrier(sycl::access::fence_space::local_space)",
    headers=["#include <CL/sycl.hpp>", "#include <iostream>"]
)

CPU_BACKEND = BackendSpec(
    name="C++ Standard (CPU)",
    kernel_macro="inline void",
    grid_macro="i", # Simple loop index
    block_macro="0",
    sync_func="// No sync needed on single thread",
    headers=["#include <vector>", "#include <iostream>", "#include <thread>"]
)

def get_all_backends():
    return [CUDA_BACKEND, HIP_BACKEND, SYCL_BACKEND, CPU_BACKEND]
