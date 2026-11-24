
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>

#define TILE 16
#define WARMUP 3
#define REPEAT 20

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ============================================================
// Padding kernel
// Inputs and output are already padded on CPU.
// A: Mplus x Nplus
// B: Nplus x Kplus
// C: Mplus x Kplus
// ============================================================

__global__ void padding_matmul(
    float *da, float *db, float *dc,
    int Mplus, int Nplus, int Kplus)
{
    __shared__ float A[16][16];
    __shared__ float B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    int circunum = Nplus / 16;
    float sum = 0.0f;

    for (int i = 0; i < circunum; i++) {

        A[threadIdx.y][threadIdx.x] =
            da[row * Nplus + i * 16 + threadIdx.x];

        B[threadIdx.y][threadIdx.x] =
            db[(i * 16 + threadIdx.y) * Kplus + col];

        __syncthreads();

        for (int j = 0; j < 16; j++) {
            sum += A[threadIdx.y][j] *
                   B[j][threadIdx.x];
        }

        __syncthreads();
    }

    dc[row * Kplus + col] = sum;
}


// ============================================================
// BC kernel
// Original design:
// 1. Full tiles in the main loop
// 2. Last incomplete tile handled separately
// ============================================================

__global__ void bc_matmul(
    float *da, float *db, float *dc,
    int M, int N, int K)
{
    __shared__ float A[16][16];
    __shared__ float B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    int circunum = (N + 15) / 16;
    int remainder = N - 16 * (circunum - 1);

    float sum = 0.0f;

    for (int i = 0; i < circunum - 1; i++) {

        int a_col = i * 16 + threadIdx.x;
        int b_row = i * 16 + threadIdx.y;

        if (row < M && a_col < N)
            A[threadIdx.y][threadIdx.x] =
                da[row * N + a_col];
        else
            A[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < N && col < K)
            B[threadIdx.y][threadIdx.x] =
                db[b_row * K + col];
        else
            B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < 16; j++)
            sum += A[threadIdx.y][j] * B[j][threadIdx.x];

        __syncthreads();
    }

    if (threadIdx.x < remainder && threadIdx.y < remainder) {
        A[threadIdx.y][threadIdx.x] =
            row < M ? da[row * N + (circunum - 1) * 16 + threadIdx.x] : 0.0f;
        B[threadIdx.y][threadIdx.x] =
            col < K ? db[((circunum - 1) * 16 + threadIdx.y) * K + col] : 0.0f;
    }
    else if (threadIdx.x < remainder && threadIdx.y >= remainder) {
        A[threadIdx.y][threadIdx.x] =
            row < M ? da[row * N + (circunum - 1) * 16 + threadIdx.x] : 0.0f;
        B[threadIdx.y][threadIdx.x] = 0.0f;
    }
    else if (threadIdx.x >= remainder && threadIdx.y < remainder) {
        A[threadIdx.y][threadIdx.x] = 0.0f;
        B[threadIdx.y][threadIdx.x] =
            col < K ? db[((circunum - 1) * 16 + threadIdx.y) * K + col] : 0.0f;
    }
    else {
        A[threadIdx.y][threadIdx.x] = 0.0f;
        B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int j = 0; j < remainder; j++)
        sum += A[threadIdx.y][j] * B[j][threadIdx.x];

    __syncthreads();

    if (row < M && col < K)
        dc[row * K + col] = sum;
}


// ============================================================
// CPU reference
// ============================================================

void cpu_matmul(
    float *a, float *b, float *c,
    int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {

            float sum = 0.0f;

            for (int k = 0; k < N; k++)
                sum += a[i * N + k] *
                       b[k * K + j];

            c[i * K + j] = sum;
        }
    }
}

bool verify(float *a, float *b, int size)
{
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        float tol = 1e-3f + 1e-3f * fabsf(b[i]);

        if (diff > tol)
            return false;
    }

    return true;
}


// ============================================================
// Padding CPU preprocessing
// This is intentionally INSIDE the timed region.
// ============================================================

void pad_inputs(
    float *a, float *b,
    float *ap, float *bp,
    int M, int N, int K,
    int Mp, int Np, int Kp)
{
    for (int i = 0; i < Mp; i++) {
        for (int j = 0; j < Np; j++) {
            if (i < M && j < N)
                ap[i * Np + j] = a[i * N + j];
            else
                ap[i * Np + j] = 0.0f;
        }
    }

    for (int i = 0; i < Np; i++) {
        for (int j = 0; j < Kp; j++) {
            if (i < N && j < K)
                bp[i * Kp + j] = b[i * K + j];
            else
                bp[i * Kp + j] = 0.0f;
        }
    }
}


// ============================================================
// One Padding end-to-end run
// CPU padding + H2D + kernel + D2H + CPU crop
// ============================================================

double run_padding(
    float *a, float *b, float *result,
    float *ap, float *bp, float *cp,
    float *da, float *db, float *dc,
    int M, int N, int K,
    int Mp, int Np, int Kp,
    LARGE_INTEGER freq)
{
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);

    // CPU preprocessing
    pad_inputs(a, b, ap, bp, M, N, K, Mp, Np, Kp);

    size_t sizea = (size_t)Mp * Np * sizeof(float);
    size_t sizeb = (size_t)Np * Kp * sizeof(float);
    size_t sizec = (size_t)Mp * Kp * sizeof(float);

    CHECK(cudaMemcpy(da, ap, sizea, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, bp, sizeb, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(Kp / 16, Mp / 16);

    padding_matmul<<<grid, block>>>(da, db, dc, Mp, Np, Kp);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(cp, dc, sizec, cudaMemcpyDeviceToHost));

    // CPU output cropping
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            result[i * K + j] = cp[i * Kp + j];
        }
    }

    QueryPerformanceCounter(&t1);

    return (double)(t1.QuadPart - t0.QuadPart)
           / freq.QuadPart * 1000.0;
}


// ============================================================
// One BC end-to-end run
// H2D + kernel + D2H
// ============================================================

double run_bc(
    float *a, float *b, float *result,
    float *da, float *db, float *dc,
    int M, int N, int K,
    LARGE_INTEGER freq)
{
    LARGE_INTEGER t0, t1;
    QueryPerformanceCounter(&t0);

    size_t sizea = (size_t)M * N * sizeof(float);
    size_t sizeb = (size_t)N * K * sizeof(float);
    size_t sizec = (size_t)M * K * sizeof(float);

    CHECK(cudaMemcpy(da, a, sizea, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, sizeb, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((K + 15) / 16, (M + 15) / 16);

    bc_matmul<<<grid, block>>>(da, db, dc, M, N, K);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(result, dc, sizec, cudaMemcpyDeviceToHost));

    QueryPerformanceCounter(&t1);

    return (double)(t1.QuadPart - t0.QuadPart)
           / freq.QuadPart * 1000.0;
}


// ============================================================
// Statistics
// ============================================================

void statistics(
    double *times, double &avg, double &stddev)
{
    avg = 0.0;

    for (int i = 0; i < REPEAT; i++)
        avg += times[i];

    avg /= REPEAT;

    double variance = 0.0;

    for (int i = 0; i < REPEAT; i++) {
        double d = times[i] - avg;
        variance += d * d;
    }

    stddev = sqrt(variance / REPEAT);
}


// ============================================================
// One test
// ============================================================

void run_test(int M, int N, int K, LARGE_INTEGER freq)
{
    int Mp = ((M + 15) / 16) * 16;
    int Np = ((N + 15) / 16) * 16;
    int Kp = ((K + 15) / 16) * 16;

    size_t sizea = (size_t)M * N * sizeof(float);
    size_t sizeb = (size_t)N * K * sizeof(float);
    size_t sizec = (size_t)M * K * sizeof(float);

    size_t sizeap = (size_t)Mp * Np * sizeof(float);
    size_t sizebp = (size_t)Np * Kp * sizeof(float);
    size_t sizecp = (size_t)Mp * Kp * sizeof(float);

    float *a = (float*)malloc(sizea);
    float *b = (float*)malloc(sizeb);
    float *ref = (float*)malloc(sizec);

    float *out_pad = (float*)malloc(sizec);
    float *out_bc = (float*)malloc(sizec);

    float *ap = (float*)malloc(sizeap);
    float *bp = (float*)malloc(sizebp);
    float *cp = (float*)malloc(sizecp);

    if (!a || !b || !ref || !out_pad || !out_bc ||
        !ap || !bp || !cp) {
        printf("Host allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < M * N; i++)
        a[i] = (float)rand() / RAND_MAX - 0.5f;

    for (int i = 0; i < N * K; i++)
        b[i] = (float)rand() / RAND_MAX - 0.5f;

    cpu_matmul(a, b, ref, M, N, K);

    float *da_pad, *db_pad, *dc_pad;
    float *da_bc, *db_bc, *dc_bc;

    CHECK(cudaMalloc((void**)&da_pad, sizeap));
    CHECK(cudaMalloc((void**)&db_pad, sizebp));
    CHECK(cudaMalloc((void**)&dc_pad, sizecp));

    CHECK(cudaMalloc((void**)&da_bc, sizea));
    CHECK(cudaMalloc((void**)&db_bc, sizeb));
    CHECK(cudaMalloc((void**)&dc_bc, sizec));

    // Warm-up both implementations
    for (int i = 0; i < WARMUP; i++) {
        run_padding(a, b, out_pad, ap, bp, cp,
                    da_pad, db_pad, dc_pad,
                    M, N, K, Mp, Np, Kp, freq);

        run_bc(a, b, out_bc, da_bc, db_bc, dc_bc,
               M, N, K, freq);
    }

    double pad_times[REPEAT];
    double bc_times[REPEAT];

    // Alternate execution order to reduce order bias
    for (int i = 0; i < REPEAT; i++) {
        if (i % 2 == 0) {
            pad_times[i] =
                run_padding(a, b, out_pad, ap, bp, cp,
                            da_pad, db_pad, dc_pad,
                            M, N, K, Mp, Np, Kp, freq);

            bc_times[i] =
                run_bc(a, b, out_bc, da_bc, db_bc, dc_bc,
                       M, N, K, freq);
        }
        else {
            bc_times[i] =
                bc_times[i] = run_bc(a, b, out_bc, da_bc, db_bc, dc_bc, M, N, K, freq);

            pad_times[i] =
                run_padding(a, b, out_pad, ap, bp, cp,
                            da_pad, db_pad, dc_pad,
                            M, N, K, Mp, Np, Kp, freq);
        }
    }

    double pad_avg, pad_std;
    double bc_avg, bc_std;

    statistics(pad_times, pad_avg, pad_std);
    statistics(bc_times, bc_avg, bc_std);

    bool pad_ok = verify(out_pad, ref, M * K);
    bool bc_ok = verify(out_bc, ref, M * K);

    printf(
        "%-4d %-4d %-4d | "
        "%-10.4f %-10.4f | "
        "%-10.4f %-10.4f | "
        "%-5s %-5s\n",
        M, N, K,
        pad_avg, pad_std,
        bc_avg, bc_std,
        pad_ok ? "PASS" : "FAIL",
        bc_ok ? "PASS" : "FAIL"
    );

    CHECK(cudaFree(da_pad));
    CHECK(cudaFree(db_pad));
    CHECK(cudaFree(dc_pad));

    CHECK(cudaFree(da_bc));
    CHECK(cudaFree(db_bc));
    CHECK(cudaFree(dc_bc));

    free(a); free(b); free(ref);
    free(out_pad); free(out_bc);
    free(ap); free(bp); free(cp);
}


// ============================================================
// Main
// ============================================================

int main()
{
    srand(42);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    int tests[][3] = {
        {2,    2,    2},
        {64,   64,   64},
        {256,  256,  256},
        {512,  512,  512},
        {1024, 1024, 1024},
        {64,   1024, 64},
        {1024, 64,   64},
        {128,  128,  2048},
        {256,  2048, 128},
        {1000, 1000, 1000}
    };

    int count = sizeof(tests) / sizeof(tests[0]);

    printf("\nFair End-to-End Comparison\n");
    printf("Timer: CPU QueryPerformanceCounter\n");
    printf("Warmup: %d | Repeat: %d\n\n", WARMUP, REPEAT);

    printf(
        "M    N    K    | Padding(ms) StdDev   | "
        "BC(ms)      StdDev   | Check\n"
    );

    printf(
        "--------------------------------------------------------------------------\n"
    );

    for (int i = 0; i < count; i++) {
        run_test(
            tests[i][0],
            tests[i][1],
            tests[i][2],
            freq
        );
    }

    return 0;
}
