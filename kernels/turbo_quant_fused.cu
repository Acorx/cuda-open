/*
 * CUDA Open - TurboQuant Fused Kernel
 * 
 * Ce kernel effectue la déquantification et la multiplication matricielle
 * en une seule passe. C'est le secret de la performance de TensorRT-LLM et vLLM.
 * 
 * Au lieu de : Dequantize(W) -> Stocker en VRAM -> MatMul
 * On fait : Charger W_int4 -> Déquantizer dans le registre -> Accumuler
 * 
 * Gain : Réduction de 50% à 75% de la bande passante mémoire utilisée.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 8

/**
 * Kernel : MatMul avec poids INT4 et Scales par groupe
 * C = (Dequantize(W_int4) * Scales) @ B
 */
__global__ void turbo_fused_matmul_int4(
    const uint8_t* __restrict__ W_int4,    // Poids compressés [M, K/2]
    const float* __restrict__ Scales,      // Scales par groupe [M, K/GroupSize]
    const float* __restrict__ B,           // Input activations [K, N]
    float* __restrict__ Output,            // Output [M, N]
    int M, int K, int N,
    int group_size
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Position globale
    int row = by * BLOCK_DIM_Y + ty;
    int col = bx * BLOCK_DIM_X + tx;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    int num_groups = K / group_size;

    // Boucle sur les groupes
    for (int g = 0; g < num_groups; g++) {
        // 1. Charger le scale du groupe (lecture mémoire rapide)
        float scale = Scales[row * num_groups + g];

        // 2. Boucle sur les éléments du groupe
        int k_start = g * group_size;
        int k_end = k_start + group_size;

        for (int k = k_start; k < k_end; k++) {
            // 3. Decoder INT4 à la volée
            // 2 poids par byte. 
            // k/2 donne l'index du byte.
            // (k%2) détermine si on prend les 4 bits de gauche ou droite.
            uint8_t packed_byte = W_int4[row * (K/2) + (k/2)];
            int int4_val;
            
            if (k % 2 == 0) {
                // 4 bits de poids faible
                int4_val = packed_byte & 0x0F;
            } else {
                // 4 bits de poids fort
                int4_val = (packed_byte >> 4) & 0x0F;
            }
            
            // Conversion INT4 -> Float (0..15 -> -8..7)
            float w_fp32 = ((float)int4_val - 8.0f) * scale;

            // 4. Accumulation directe (FMA)
            sum += w_fp32 * B[k * N + col];
        }
    }

    // 5. Écriture unique en mémoire globale
    Output[row * N + col] = sum;
}

// ============================================================================
// API C pour Python
// ============================================================================

extern "C" {

int launch_turbo_fused_int4(
    const uint8_t* W_int4,
    const float* Scales,
    const float* B,
    float* Output,
    int M, int K, int N,
    int group_size
) {
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X, 
              (M + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    turbo_fused_matmul_int4<<<grid, block>>>(W_int4, Scales, B, Output, M, K, N, group_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "TurboQuant Kernel Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    return 0;
}

} // extern "C"
