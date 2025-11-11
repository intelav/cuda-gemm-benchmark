/******************************************************************************************
 * matmul_variants_shmoo.cu
 *
 * Performs Shmoo-style benchmarking for multiple GEMM (matrix multiplication) variants:
 *   0️⃣ FP32 Shared-memory GEMM
 *   1️⃣ Tensor Core GEMM (FP16→FP32)
 *   2️⃣ Tensor Core GEMM (BF16→FP32)
 *
 * It sweeps matrix sizes (e.g., 256 → 4096), measures runtime, throughput, and accuracy,
 * and logs results to CSV for later visualization.
 *
 * Author: ChatGPT (OpenAI)
 * Date: 2025
 ******************************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ------------------------------------------------------------------------------------------------
// Shared Memory FP32 GEMM
// ------------------------------------------------------------------------------------------------

#define TILE_SIZE 16

__global__ void matmul_sharedmem_kernel(const float *A, const float *B, float *C,
                                        int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];
    float value = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && (t * TILE_SIZE + tx) < K)
            Asub[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            Asub[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < K && col < N)
            Bsub[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bsub[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            value += Asub[ty][k] * Bsub[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = value;
}

// ------------------------------------------------------------------------------------------------
// Tensor Core GEMM (FP16→FP32)
// ------------------------------------------------------------------------------------------------

__global__ void matmul_tensorcore_fp16_kernel(const half *A, const half *B, float *C,
                                              int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += 16) {
        const half *tileA = A + (warpM * 16) * K + i;
        const half *tileB = B + (i) * N + (warpN * 16);
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float *tileC = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}

// ------------------------------------------------------------------------------------------------
// Tensor Core GEMM (BF16→FP32)
// ------------------------------------------------------------------------------------------------

__global__ void matmul_tensorcore_bf16_kernel(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
                                              int M, int N, int K) {
#if __CUDA_ARCH__ >= 800
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);
    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += 16) {
        const __nv_bfloat16 *tileA = A + (warpM * 16) * K + i;
        const __nv_bfloat16 *tileB = B + (i) * N + (warpN * 16);
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float *tileC = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("BF16 Tensor Core requires sm_80+\n");
#endif
}

// ------------------------------------------------------------------------------------------------
// CPU reference GEMM
// ------------------------------------------------------------------------------------------------

void cpu_matmul(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C,
                int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ------------------------------------------------------------------------------------------------
// Time measurement
// ------------------------------------------------------------------------------------------------

float measure_gemm_time(dim3 grid, dim3 block, int variant,
                        const void *A, const void *B, float *C,
                        int M, int N, int K) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    switch (variant) {
        case 0:
            matmul_sharedmem_kernel<<<grid, block>>>((const float*)A, (const float*)B, C, M, N, K);
            break;
        case 1:
            matmul_tensorcore_fp16_kernel<<<grid, block>>>((const half*)A, (const half*)B, C, M, N, K);
            break;
        case 2:
            matmul_tensorcore_bf16_kernel<<<grid, block>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B, C, M, N, K);
            break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ------------------------------------------------------------------------------------------------
// Main: Shmoo Sweep Benchmark
// ------------------------------------------------------------------------------------------------

int main() {
    std::ofstream csv("shmoo_results.csv");
    csv << "Variant,MatrixSize,Time_ms,GFLOPs,MaxAbsError\n";

    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};

    for (int N : sizes) {
        int M = N, K = N;
        std::cout << "\n=== Matrix Size: " << N << "x" << N << " ===\n";

        // Host memory
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
        for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

        // Device allocations
        float *d_A32, *d_B32, *d_C;
        half *d_A16, *d_B16;
        __nv_bfloat16 *d_Abf16, *d_Bbf16;
        cudaMalloc(&d_A32, M * K * sizeof(float));
        cudaMalloc(&d_B32, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMalloc(&d_A16, M * K * sizeof(half));
        cudaMalloc(&d_B16, K * N * sizeof(half));
        cudaMalloc(&d_Abf16, M * K * sizeof(__nv_bfloat16));
        cudaMalloc(&d_Bbf16, K * N * sizeof(__nv_bfloat16));

        cudaMemcpy(d_A32, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B32, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A16, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B16, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Abf16, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bbf16, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        for (int variant = 0; variant < 3; variant++) {
            const char *name =
                (variant == 0) ? "FP32 Shared Memory GEMM" :
                (variant == 1) ? "Tensor Core GEMM (FP16→FP32)" :
                                 "Tensor Core GEMM (BF16→FP32)";
            std::cout << "Running " << name << "...\n";

            float ms = measure_gemm_time(grid, threads, variant,
                                         (variant == 0 ? (void*)d_A32 :
                                          variant == 1 ? (void*)d_A16 :
                                                         (void*)d_Abf16),
                                         (variant == 0 ? (void*)d_B32 :
                                          variant == 1 ? (void*)d_B16 :
                                                         (void*)d_Bbf16),
                                         d_C, M, N, K);

            double flops = 2.0 * M * N * K;
            double gflops = flops / (ms * 1.0e6);

            cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            cpu_matmul(h_A, h_B, h_ref, M, N, K);

            float max_err = 0.0f;
            for (int i = 0; i < M * N; i++)
                max_err = std::max(max_err, fabs(h_ref[i] - h_C[i]));

            std::cout << "  Time: " << ms << " ms,  GFLOPs: " << gflops
                      << ",  MaxErr: " << max_err << "\n";

            csv << variant << "," << N << "," << ms << "," << gflops << "," << max_err << "\n";
        }

        cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C);
        cudaFree(d_A16); cudaFree(d_B16);
        cudaFree(d_Abf16); cudaFree(d_Bbf16);
    }

    csv.close();
    std::cout << "\n✅ Shmoo benchmarking completed. Results saved to shmoo_results.csv\n";
    return 0;
}
