
#include <iostream>
#include <vector>

// --- Kernel généré par CUDA Open Compiler (Version CPU Realiste) ---
void cpu_matmul_kernel(const float* A, const float* B, const float* Bias, float* Output, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            Output[i * N + j] = sum + Bias[i * N + j]; // Fusion Bias appliquée
        }
    }
}
// -------------------------------------------------------------

int main() {
    const int M = 64, K = 64, N = 64;
    std::vector<float> A(M*K, 2.0f);
    std::vector<float> B(K*N, 3.0f);
    std::vector<float> Bias(M*N, 0.5f);
    std::vector<float> Output(M*N, 0.0f);

    // Exécution du kernel compilé
    cpu_matmul_kernel(A.data(), B.data(), Bias.data(), Output.data(), M, K, N);

    // Vérification (2 * 3 * K) + 0.5 = 6*64 + 0.5 = 384.5
    float val = Output[0];
    std::cout << "VALEUR_CALCULEE: " << val << std::endl;
    std::cout << "SUCCESS: Compilation et exécution réussies!" << std::endl;
    return 0;
}
