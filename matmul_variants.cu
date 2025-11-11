/**********************************************************************************************
 * matmul_variants.cu
 *
 * This file implements two versions of matrix multiplication (GEMM):
 *
 *  (1) matmul_shared_kernel()  — Standard FP32 GEMM using shared memory and CUDA cores.
 *  (2) matmul_tensorcore_kernel() — Mixed-precision GEMM using Tensor Cores (WMMA API).
 *
 *  The program benchmarks both variants and compares their performance.
 *
 *  Compile:
 *      nvcc matmul_variants.cu -O3 --gpu-architecture=sm_80 -o matmul_variants
 *
 *  Run:
 *      ./matmul_variants
 *
 *  Author: ChatGPT (2025)
 **********************************************************************************************/

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector> 
using namespace nvcuda;

// ==============================================================================================
// Kernel 1 — Shared Memory FP32 GEMM (using standard CUDA cores)
// ==============================================================================================

#define TILE_SIZE 16  // Tile dimension (16×16 threads per block)

// Classic GEMM kernel using shared memory blocking and FP32 CUDA cores
__global__ void matmul_shared_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K)
{
    // Shared memory tiles for submatrices
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Compute global row/col for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over all tiles of the K dimension (the “shared” inner dimension of A and B)
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

    // ------------------------------------------------------------------------
    // Load one TILE_SIZE × TILE_SIZE block of A and B into shared memory
    // ------------------------------------------------------------------------
    //
    // Each thread in the block cooperatively loads a single element of A and B
    // into a shared-memory tile (Asub and Bsub).
    // The tile index `t` determines which “slice” of K we are processing.
    //
    // For matrix multiplication:  C = A × B
    //
    //   A is of size M×K
    //   B is of size K×N
    //   C is of size M×N
    //
    // The multiplication is done by summing across the K dimension in chunks
    // of size TILE_SIZE at a time. Each iteration processes one chunk.
    // ------------------------------------------------------------------------

    // Load element of A into shared memory (if within bounds)
    // Each thread loads one element A[row, t*TILE_SIZE + threadIdx.x]
    // - 'row' identifies the row of A this thread is responsible for.
    // - 't*TILE_SIZE + threadIdx.x' picks the correct column of A for this tile.
    // If we're outside the matrix bounds (when K is not divisible by TILE_SIZE),
    // set the element to 0 to avoid invalid memory access.row*k is stride.
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
        Asub[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    else
        Asub[threadIdx.y][threadIdx.x] = 0.0f;

    // Load element of B into shared memory (if within bounds)
    // Each thread loads one element B[t*TILE_SIZE + threadIdx.y, col]
    // - 't*TILE_SIZE + threadIdx.y' gives the correct row of B for this tile.
    // - 'col' identifies which column of B this thread is working on.
    //hence (t * TILE_SIZE + threadIdx.y) * N  is the stride here.
    // Again, if the tile goes beyond matrix bounds, load 0.
    if ((t * TILE_SIZE + threadIdx.y) < K && col < N)
        Bsub[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    else
        Bsub[threadIdx.y][threadIdx.x] = 0.0f;

    // Ensure all threads have finished loading their elements
    // before starting computation. Shared memory tiles Asub and Bsub
    // must be fully populated before any thread uses them.
    __syncthreads();

    // ------------------------------------------------------------------------
    // Compute partial results for this tile (matrix multiply of Asub × Bsub)
    // ------------------------------------------------------------------------
    //
    // Each thread computes one element of the C tile:
    //    C[row, col] = sum_k (A[row, k] * B[k, col])
    //
    // Since we only have TILE_SIZE elements per tile along K,
    // this loop accumulates partial sums over that small subset.
    // These partial sums will be combined over multiple tiles in t-loop.
    //
    // The innermost loop iterates over the k-dimension *within the tile*.
    // Each thread multiplies Asub[row_in_tile, k] * Bsub[k, col_in_tile]
    // and accumulates into its local register variable `value`.
    // ------------------------------------------------------------------------
    for (int k = 0; k < TILE_SIZE; k++)
        value += Asub[threadIdx.y][k] * Bsub[k][threadIdx.x];

    // Synchronize again before loading the next tile
    // (to avoid overwriting Asub/Bsub while other threads are still using them)
    __syncthreads();
    }


    // Write the result back to C
    if (row < M && col < N)
        C[row * N + col] = value;
}

// ==============================================================================================
// Kernel 2 — Tensor Core GEMM using WMMA API (FP16 input, FP32 accumulate)
// ==============================================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ============================================================================
// Tensor Core GEMM kernel (FP16 × FP16 → FP32 accumulation)
//
// Each warp computes one 16×16 tile of the output matrix C using Tensor Cores.
// This kernel uses the WMMA API (Warp Matrix Multiply and Accumulate) provided
// in <mma.h>, available on Volta and later architectures (sm_70+).
//
// The input matrices A and B are in half precision (FP16).
// The output/accumulator matrix C is in single precision (FP32).
// ============================================================================

__global__ void matmul_tensorcore_kernel(const half *A, const half *B, float *C,
                                         int M, int N, int K)
{
    using namespace nvcuda;

    // ------------------------------------------------------------------------
    // (1) Compute the warp indices in the overall grid.
    //
    // Each block may contain multiple warps (typically 4 warps for dim3(2,2)),
    // and each warp is responsible for computing a single 16×16 tile of C.
    //
    // - warpM: which tile of C in the vertical direction (rows)
    // - warpN: which tile of C in the horizontal direction (cols)
    //
    // Each tile corresponds to a 16×16 region of the output matrix.
    // ------------------------------------------------------------------------
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // If this warp’s tile would go out of bounds (beyond M or N), skip it.
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) 
        return;

    // ------------------------------------------------------------------------
    // (2) Declare WMMA fragments.
    //
    // A fragment represents a small matrix tile (16×16×16 here) that lives
    // in registers and can be multiplied by Tensor Cores in a single operation.
    //
    // - a_frag : fragment of matrix A (input)
    // - b_frag : fragment of matrix B (input)
    // - c_frag : fragment of matrix C (accumulator, output)
    //
    // Layouts:
    //   A → row-major (standard C memory layout)
    //   B → column-major (optimized for Tensor Core memory access)
    //
    // Accumulator (C) always stored as row-major float values.
    // ------------------------------------------------------------------------
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize C fragment to 0 before accumulation.
    wmma::fill_fragment(c_frag, 0.0f);

    // ------------------------------------------------------------------------
    // (3) Iterate over the K dimension in chunks of 16 elements.
    //
    // For each iteration:
    //  - Load one 16×16 tile from A (row-major)
    //  - Load one 16×16 tile from B (col-major)
    //  - Perform a Tensor Core matrix multiply-accumulate:
    //        C_tile += A_tile × B_tile
    //
    // The loop runs K/16 times since each Tensor Core operation processes
    // 16 elements of K per step (WMMA_K = 16).
    // ------------------------------------------------------------------------
    for (int i = 0; i < K; i += WMMA_K) {

        // ------------------------------------------------------------
        // Compute memory offsets for A and B tiles.
        //
        // A tile starts at:
        //   row = warpM * 16
        //   col = i (K offset for this iteration)
        //   (warpM * WMMA_M) is the stride to get to the correct row in A.
        // B tile starts at:
        //   row = i (same K offset)
        //   col = warpN * 16
        //  (warpN * WMMA_N) is the stride to get to the correct column in B.
        // These base pointers ensure each warp loads the correct
        // region of A and B from global memory.
        // ------------------------------------------------------------
        const half *tileA = A + (warpM * WMMA_M) * K + i;     // A[row_offset, i]
        const half *tileB = B + (i) * N + (warpN * WMMA_N);   // B[i, col_offset]

        // ------------------------------------------------------------
        // Load these submatrices into WMMA fragments (into registers).
        //
        // The third argument is the leading dimension:
        //   - For A, it’s K (stride between consecutive rows)
        //   - For B, it’s N (stride between consecutive rows in col-major)
        //
        // These calls automatically handle the fragment mapping from
        // global memory to the per-warp registers used by Tensor Cores.
        // ------------------------------------------------------------
        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);

        // ------------------------------------------------------------
        // Perform the matrix multiply-accumulate operation on Tensor Cores:
        //
        //   c_frag = a_frag × b_frag + c_frag
        //
        // Each warp executes one Tensor Core instruction, which internally
        // performs 16×16×16 fused multiply-adds (256 FMAs) in one cycle.
        //
        // Results accumulate into the floating-point accumulator fragment.
        // ------------------------------------------------------------
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ------------------------------------------------------------------------
    // (4) Store the computed C tile back to global memory.
    //
    // The fragment `c_frag` now contains one 16×16 block of the output matrix.
    // We compute the base address for this tile in C and store it in row-major.
    // ------------------------------------------------------------------------
    float *tileC = C + (warpM * WMMA_M) * N + (warpN * WMMA_N);
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
}


// ==============================================================================================
// Helper functions
// ==============================================================================================

// ----------------------------------------------------------------------------------------------
// CPU reference implementation of GEMM (for correctness checking)
// Computes: C = A × B
// A[M×K], B[K×N], C[M×N]
// ----------------------------------------------------------------------------------------------
void matmul_cpu_ref(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Compare GPU result with CPU reference
// Returns maximum absolute difference
// ----------------------------------------------------------------------------------------------
float compare_results(const float *ref, const float *gpu, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        max_err = fmaxf(max_err, fabsf(ref[i] - gpu[i]));
    }
    return max_err;
}

void fill_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// ==============================================================================================
// Host launcher and benchmarking
// ==============================================================================================

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    std::cout << "=== Matrix Multiplication Variants ===\n";
    std::cout << "Matrix sizes: " << M << " x " << K << " * " << K << " x " << N << "\n\n";

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Host allocations
    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C = (float *)malloc(bytes_C);
    float *h_C_ref = (float *)malloc(bytes_C);

    fill_matrix(h_A, M * K);
    fill_matrix(h_B, K * N);

    // Device allocations (FP32)
    float *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, bytes_A), "malloc A");
    check_cuda(cudaMalloc(&d_B, bytes_B), "malloc B");
    check_cuda(cudaMalloc(&d_C, bytes_C), "malloc C");

    check_cuda(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice), "memcpy A");
    check_cuda(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice), "memcpy B");

    // ------------------------------------------------------------
    // Variant 1 — FP32 Shared-Memory GEMM
    // ------------------------------------------------------------
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Running FP32 Shared Memory GEMM...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    matmul_shared_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    double time_ms1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "  Time: " << time_ms1 << " ms\n";

    double gflops1 = 2.0 * M * N * K / (time_ms1 * 1e6);
    std::cout << "  Throughput: " << gflops1 << " GFLOP/s\n\n";
        
    // Copy back result
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Compute reference on CPU
    matmul_cpu_ref(h_A, h_B, h_C_ref, M, N, K);

    // Compare
    float max_err1 = compare_results(h_C_ref, h_C, M * N);
    std::cout << "  Max absolute error (vs CPU): " << max_err1 << "\n";

    // ------------------------------------------------------------
    // Variant 2 — Tensor Core GEMM
    // ------------------------------------------------------------
    half *d_Ah, *d_Bh;
    check_cuda(cudaMalloc(&d_Ah, M * K * sizeof(half)), "malloc A_h");
    check_cuda(cudaMalloc(&d_Bh, K * N * sizeof(half)), "malloc B_h");

    // Convert host FP32 to FP16
    std::vector<half> h_Ah(M * K), h_Bh(K * N);
    for (int i = 0; i < M * K; i++) h_Ah[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_Bh[i] = __float2half(h_B[i]);
        
    std::vector<float> h_B_col(K * N);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B_col[i + j * K] = h_B[i * N + j];  // transpose

    // Then convert to half and copy
    for (int i = 0; i < K * N; ++i)
        h_Bh[i] = __float2half(h_B_col[i]);

    cudaMemcpy(d_Bh, h_Bh.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    check_cuda(cudaMemcpy(d_Ah, h_Ah.data(), M * K * sizeof(half), cudaMemcpyHostToDevice), "memcpy A_h");
    check_cuda(cudaMemcpy(d_Bh, h_Bh.data(), K * N * sizeof(half), cudaMemcpyHostToDevice), "memcpy B_h");

    dim3 threads_tc(2, 2);  // 4 warps per block
    dim3 blocks_tc((N + WMMA_N - 1) / WMMA_N,
                   (M + WMMA_M - 1) / WMMA_M);

    std::cout << "Running Tensor Core GEMM (FP16→FP32)...\n";
    auto t3 = std::chrono::high_resolution_clock::now();
    matmul_tensorcore_kernel<<<blocks_tc, threads_tc>>>(d_Ah, d_Bh, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    double time_ms2 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "  Time: " << time_ms2 << " ms\n";

    double gflops2 = 2.0 * M * N * K / (time_ms2 * 1e6);
    std::cout << "  Throughput: " << gflops2 << " GFLOP/s\n\n";

    // Copy back result from Tensor Core kernel
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Compare GPU Tensor Core result with CPU reference
    float max_err2 = compare_results(h_C_ref, h_C, M * N);
    std::cout << "  Max absolute error (vs CPU): " << max_err2 << "\n";

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Ah);
    cudaFree(d_Bh);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    std::cout << "✅ Done.\n";
    return 0;
}
