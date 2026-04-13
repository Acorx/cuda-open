
#include <iostream>
#include <vector>
#include <cmath>

// --- Logique générée par le compilateur CUDA Open (Fusion MatMul+Bias) ---
void run_kernel(const float* A, const float* B, const float* Bias, float* Out, int M, int K, int N) {
    // Implémentation séquentielle pour CPU (générée depuis l'IR optimisé)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Simulation de la déquantification INT4 -> Float (TurboQuant)
                // Dans un vrai kernel GPU, cela se fait dans les registres
                float w = B[k * N + j]; // On utilise les poids FP32 ici pour la démo CPU
                sum += A[i * K + k] * w;
            }
            // Fusion de l'ajout du Bias (optimisation détectée par le compilateur)
            Out[i * N + j] = sum + Bias[i * N + j];
        }
    }
}

int main() {
    const int M = 128, K = 128, N = 128;
    std::vector<float> A(M*K), B(K*N), Bias(M*N), Out(M*N);

    // On remplit avec des valeurs connues pour vérifier
    for(int i=0; i<M*K; ++i) A[i] = 0.5f;
    for(int i=0; i<K*N; ++i) B[i] = 2.0f;
    for(int i=0; i<M*N; ++i) Bias[i] = 0.1f;

    // Exécution du kernel compilé
    run_kernel(A.data(), B.data(), Bias.data(), Out.data(), M, K, N);

    // Calcul théorique attendu : (0.5 * 2.0 * K) + 0.1 = K + 0.1
    float expected = (0.5f * 2.0f * K) + 0.1f;
    float actual = Out[0];

    std::cout << "RESULTAT_C++: " << actual << std::endl;
    std::cout << "ATTENDU: " << expected << std::endl;
    
    if (std::abs(actual - expected) < 0.001f) {
        std::cout << "STATUS: SUCCESS" << std::endl;
    } else {
        std::cout << "STATUS: FAIL" << std::endl;
    }
    return 0;
}
