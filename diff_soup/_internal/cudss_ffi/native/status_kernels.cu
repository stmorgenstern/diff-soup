// The per-lane status scan of cuDSS factors held for the phased factor contract (see
// status_kernels.h): one kernel reads every value and every factor diagonal entry (or status
// solve error) of a batch of lanes once and reduces, per lane, a non-finite flag for the values,
// a non-finite flag for the diagonal and the smallest and largest finite diagonal magnitude; the
// right-hand sides and errors of a batch factor's status solve; and the two verdict kernels of a
// single factor (QueueValuesVerdict, QueueSolveVerdict), which reduce across their blocks and let
// the last block to finish write the result.
//
// Each lane is split into the same number of blocks; a block reduces its part in registers,
// across its warps in shared memory, and leaves one atomic update per field in the lane's
// result. Magnitudes are reduced as the bits of their float64 value (non-negative IEEE numbers
// order like their bit patterns): the largest with an atomic maximum, the smallest with an
// atomic maximum of the complement, so that a zeroed result is the identity of both.

#include "complex_scalar.cuh"
#include "status_kernels.h"

namespace diff_soup_status {
namespace {

using diff_soup_complex::Complex;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// Entries each thread reads per lane before more blocks are given to the lane.
constexpr int64_t kEntriesPerThread = 8;
constexpr int64_t kMaxBlocksPerLane = 1024;
constexpr int64_t kMaxBlocks = 2147483647;

__device__ __forceinline__ bool NonFinite(double x) {
    return (static_cast<unsigned long long>(__double_as_longlong(x)) &
            0x7ff0000000000000ull) == 0x7ff0000000000000ull;
}

__device__ __forceinline__ bool NonFinite(float x) {
    return (__float_as_uint(x) & 0x7f800000u) == 0x7f800000u;
}

__device__ __forceinline__ unsigned long long MagnitudeBits(double x) {
    return static_cast<unsigned long long>(__double_as_longlong(fabs(x)));
}

__device__ __forceinline__ unsigned long long MagnitudeBits(float x) {
    return static_cast<unsigned long long>(__double_as_longlong(fabs(static_cast<double>(x))));
}

// A complex entry is non-finite when either part is; its magnitude is the modulus.
template <typename T>
__device__ __forceinline__ bool NonFinite(const Complex<T>& z) {
    return NonFinite(z.re) || NonFinite(z.im);
}

template <typename T>
__device__ __forceinline__ unsigned long long MagnitudeBits(const Complex<T>& z) {
    return static_cast<unsigned long long>(
        __double_as_longlong(hypot(static_cast<double>(z.re), static_cast<double>(z.im))));
}

// `values` are scanned entry by entry as T (a complex value as its two parts); `diagonal`
// entries are of type D (T, or Complex<T>, whose magnitude is the modulus).
template <typename T, typename D = T>
__global__ void __launch_bounds__(kThreads)
    LaneStatusKernel(const T* __restrict__ values, int64_t value_count,
                     const D* __restrict__ diagonal, int64_t diagonal_count,
                     int64_t blocks_per_lane, LaneStatusWords* __restrict__ out) {
    const int64_t lane = static_cast<int64_t>(blockIdx.x) / blocks_per_lane;
    const int64_t part = static_cast<int64_t>(blockIdx.x) - lane * blocks_per_lane;
    const int64_t stride = blocks_per_lane * kThreads;
    const int64_t start = part * kThreads + threadIdx.x;

    bool values_bad = false;
    if (values != nullptr) {
        const T* v = values + lane * value_count;
#pragma unroll 4
        for (int64_t i = start; i < value_count; i += stride) values_bad |= NonFinite(v[i]);
    }
    bool diagonal_bad = false;
    unsigned long long smallest = ~0ull;  // above the bits of every finite magnitude
    unsigned long long largest = 0ull;
    if (diagonal != nullptr) {
        const D* d = diagonal + lane * diagonal_count;
#pragma unroll 4
        for (int64_t i = start; i < diagonal_count; i += stride) {
            const D x = d[i];
            if (NonFinite(x)) {
                diagonal_bad = true;
            } else {
                const unsigned long long bits = MagnitudeBits(x);
                smallest = bits < smallest ? bits : smallest;
                largest = bits > largest ? bits : largest;
            }
        }
    }

    values_bad = __syncthreads_or(values_bad) != 0;
    diagonal_bad = __syncthreads_or(diagonal_bad) != 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long s = __shfl_down_sync(0xffffffffu, smallest, offset);
        const unsigned long long l = __shfl_down_sync(0xffffffffu, largest, offset);
        smallest = s < smallest ? s : smallest;
        largest = l > largest ? l : largest;
    }
    __shared__ unsigned long long warp_smallest[kWarps];
    __shared__ unsigned long long warp_largest[kWarps];
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        warp_smallest[warp] = smallest;
        warp_largest[warp] = largest;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int w = 1; w < kWarps; ++w) {
        smallest = warp_smallest[w] < smallest ? warp_smallest[w] : smallest;
        largest = warp_largest[w] > largest ? warp_largest[w] : largest;
    }
    LaneStatusWords* result = out + lane;
    if (values_bad) atomicOr(&result->values_nonfinite, 1u);
    if (diagonal_bad) atomicOr(&result->diagonal_nonfinite, 1u);
    if (smallest != ~0ull) {
        atomicMax(reinterpret_cast<unsigned long long*>(&result->smallest_complement),
                  ~smallest);
    }
    if (largest != 0ull) {
        atomicMax(reinterpret_cast<unsigned long long*>(&result->largest), largest);
    }
}

// Blocks of a verdict kernel over `count` entries: about kEntriesPerThread entries per thread,
// at most kMaxBlocksPerLane blocks, at least one.
unsigned int VerdictBlocks(int64_t count) {
    const int64_t per_block = kThreads * kEntriesPerThread;
    int64_t blocks = (count + per_block - 1) / per_block;
    if (blocks > kMaxBlocksPerLane) blocks = kMaxBlocksPerLane;
    if (blocks < 1) blocks = 1;
    return static_cast<unsigned int>(blocks);
}

// Every block ORs its finding into words->flags and counts itself in words->arrived; the last
// block to arrive reads the flags (every earlier block's update is visible after its fence),
// sets the status and returns both scratch words to zero for the next launch.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    ValuesVerdictKernel(const T* __restrict__ values, int64_t count,
                        SideVerdictWords* __restrict__ words, VerdictResult result) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    bool bad = false;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < count;
         i += stride) {
        bad |= NonFinite(values[i]);
    }
    bad = __syncthreads_or(bad) != 0;
    if (threadIdx.x != 0) return;
    if (bad) atomicOr(&words->flags, 1u);
    __threadfence();
    if (atomicAdd(&words->arrived, 1u) != gridDim.x - 1) return;
    const unsigned int flags = atomicExch(&words->flags, 0u);
    words->arrived = 0u;
    const int32_t status = flags != 0u ? kVerdictNonfinite : kVerdictSuccess;
    words->status = status;
    for (int i = 0; i < result.count; ++i) {
        *result.targets[i] = i == result.status_index ? status : result.values[i];
    }
}

// As ValuesVerdictKernel, over a solve's right-hand sides (bit 1) and solution (bit 2); the last
// block also writes the solve's status and, when the factor failed, nan into every solution
// entry.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    SolveVerdictKernel(const T* __restrict__ b, T* __restrict__ x, int64_t total,
                       SideVerdictWords* __restrict__ words, int32_t* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    bool b_bad = false;
    bool x_bad = false;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total;
         i += stride) {
        b_bad |= NonFinite(b[i]);
        x_bad |= NonFinite(x[i]);
    }
    b_bad = __syncthreads_or(b_bad) != 0;
    x_bad = __syncthreads_or(x_bad) != 0;
    __shared__ int32_t verdict;  // the solve's status in the last block, -1 in every other
    if (threadIdx.x == 0) {
        verdict = -1;
        const unsigned int mine = (b_bad ? 1u : 0u) | (x_bad ? 2u : 0u);
        if (mine != 0u) atomicOr(&words->flags, mine);
        __threadfence();
        if (atomicAdd(&words->arrived, 1u) == gridDim.x - 1) {
            const unsigned int flags = atomicExch(&words->flags, 0u);
            words->arrived = 0u;
            int32_t status = words->status;
            if (status == kVerdictSuccess && flags == 2u) {
                status = kVerdictSingular;
                words->status = status;
            }
            words->solve_status = status;
            if (out != nullptr) *out = status;
            verdict = status;
        }
    }
    __syncthreads();
    if (verdict <= kVerdictSuccess) return;
    const T nan = static_cast<T>(__longlong_as_double(0x7ff8000000000000ll));
    for (int64_t i = threadIdx.x; i < total; i += kThreads) x[i] = nan;
}

// x*_i of the status solve (status_kernels.h).
template <typename T>
__device__ __forceinline__ T ProbeSolution(int64_t i) {
    return static_cast<T>(1) + static_cast<T>(i % 7) * static_cast<T>(0.125);
}

// One thread per row of a lane, grid-strided over every row of every lane. V is the value type
// (T, or Complex<T>, whose x* is real).
template <typename T, typename V = T>
__global__ void __launch_bounds__(kThreads)
    ProbeRightHandSidesKernel(const int32_t* __restrict__ offsets,
                              const int32_t* __restrict__ columns, const V* __restrict__ values,
                              int64_t n, int64_t nse, int64_t total, V* __restrict__ b) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total;
         k += stride) {
        const int64_t lane = k / n;
        const int64_t row = k - lane * n;
        const V* v = values + lane * nse;
        V sum = static_cast<V>(0);
        for (int32_t p = offsets[row]; p < offsets[row + 1]; ++p) {
            sum += v[p] * ProbeSolution<T>(columns[p]);
        }
        b[k] = sum;
    }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    ProbeErrorsKernel(T* __restrict__ x, int64_t n, int64_t total) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total;
         k += stride) {
        x[k] -= ProbeSolution<T>(k % n);
    }
}

// The complex error x - x*: x* is real, so only the real part moves.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    ProbeErrorsKernel(Complex<T>* __restrict__ x, int64_t n, int64_t total) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t k = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; k < total;
         k += stride) {
        x[k].re -= ProbeSolution<T>(k % n);
    }
}

// Blocks of a grid-strided kernel over `total` entries: one entry per thread, at most 65,535
// blocks, at least one.
unsigned int StridedBlocks(int64_t total) {
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    return static_cast<unsigned int>(blocks);
}

// Calls `launch(real, scalar)` with a value of the real type of a factor's width (double for
// float64 and complex128, float otherwise) and of its scalar type (that real type, or Complex
// of it), the template arguments of a queue function's kernels.
template <typename Launch>
void DispatchScalar(bool f64, bool complex, Launch&& launch) {
    if (complex && f64) {
        launch(double{}, Complex<double>{});
    } else if (complex) {
        launch(float{}, Complex<float>{});
    } else if (f64) {
        launch(double{}, double{});
    } else {
        launch(float{}, float{});
    }
}

}  // namespace

cudaError_t QueueLaneStatusScan(cudaStream_t stream, bool f64, const void* values,
                                int64_t value_count, const void* diagonal,
                                int64_t diagonal_count, int64_t lanes, LaneStatusWords* out,
                                bool complex) {
    if (lanes <= 0) return cudaSuccess;
    if (lanes > kMaxBlocks) return cudaErrorInvalidValue;
    cudaError_t err = cudaMemsetAsync(out, 0, static_cast<size_t>(lanes) * sizeof(*out), stream);
    if (err != cudaSuccess) return err;
    if (values == nullptr || value_count <= 0) {
        values = nullptr;
        value_count = 0;
    }
    if (diagonal == nullptr || diagonal_count <= 0) {
        diagonal = nullptr;
        diagonal_count = 0;
    }
    const int64_t longest = value_count > diagonal_count ? value_count : diagonal_count;
    if (longest == 0) return cudaSuccess;
    const int64_t per_block = kThreads * kEntriesPerThread;
    int64_t blocks_per_lane = (longest + per_block - 1) / per_block;
    if (blocks_per_lane > kMaxBlocksPerLane) blocks_per_lane = kMaxBlocksPerLane;
    if (blocks_per_lane > kMaxBlocks / lanes) blocks_per_lane = kMaxBlocks / lanes;
    if (blocks_per_lane < 1) blocks_per_lane = 1;
    const dim3 grid(static_cast<unsigned int>(lanes * blocks_per_lane));
    // Each complex value is scanned as its two parts: a lane holds 2 * value_count of them.
    const int64_t parts = complex ? 2 : 1;
    DispatchScalar(f64, complex, [&](auto real, auto scalar) {
        using R = decltype(real);
        using S = decltype(scalar);
        LaneStatusKernel<R, S><<<grid, kThreads, 0, stream>>>(
            static_cast<const R*>(values), parts * value_count, static_cast<const S*>(diagonal),
            diagonal_count, blocks_per_lane, out);
    });
    return cudaGetLastError();
}

cudaError_t QueueValuesVerdict(cudaStream_t stream, bool f64, const void* values, int64_t count,
                               SideVerdictWords* words, const VerdictResult* result) {
    if (words == nullptr || count < 0 || (count > 0 && values == nullptr) ||
        (result != nullptr && (result->count < 0 || result->count > VerdictResult::kMax))) {
        return cudaErrorInvalidValue;
    }
    const VerdictResult none;
    const VerdictResult& write = result != nullptr ? *result : none;
    const unsigned int blocks = VerdictBlocks(count);
    if (f64) {
        ValuesVerdictKernel<double><<<blocks, kThreads, 0, stream>>>(
            static_cast<const double*>(values), count, words, write);
    } else {
        ValuesVerdictKernel<float><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(values), count, words, write);
    }
    return cudaGetLastError();
}

cudaError_t QueueSolveVerdict(cudaStream_t stream, bool f64, const void* b, void* x,
                              int64_t total, SideVerdictWords* words, int32_t* out) {
    if (words == nullptr || total < 0 || (total > 0 && (b == nullptr || x == nullptr))) {
        return cudaErrorInvalidValue;
    }
    const unsigned int blocks = VerdictBlocks(total);
    if (f64) {
        SolveVerdictKernel<double><<<blocks, kThreads, 0, stream>>>(
            static_cast<const double*>(b), static_cast<double*>(x), total, words, out);
    } else {
        SolveVerdictKernel<float><<<blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(b), static_cast<float*>(x), total, words, out);
    }
    return cudaGetLastError();
}

cudaError_t QueueProbeRightHandSides(cudaStream_t stream, bool f64, const int32_t* offsets,
                                     const int32_t* columns, const void* values, int64_t n,
                                     int64_t nse, int64_t lanes, void* b, bool complex) {
    if (n <= 0 || lanes <= 0) return cudaSuccess;
    if (offsets == nullptr || columns == nullptr || values == nullptr || b == nullptr) {
        return cudaErrorInvalidValue;
    }
    const int64_t total = n * lanes;
    const unsigned int blocks = StridedBlocks(total);
    DispatchScalar(f64, complex, [&](auto real, auto scalar) {
        using R = decltype(real);
        using S = decltype(scalar);
        ProbeRightHandSidesKernel<R, S><<<blocks, kThreads, 0, stream>>>(
            offsets, columns, static_cast<const S*>(values), n, nse, total, static_cast<S*>(b));
    });
    return cudaGetLastError();
}

cudaError_t QueueProbeErrors(cudaStream_t stream, bool f64, void* x, int64_t n, int64_t lanes,
                             bool complex) {
    if (n <= 0 || lanes <= 0) return cudaSuccess;
    if (x == nullptr) return cudaErrorInvalidValue;
    const int64_t total = n * lanes;
    const unsigned int blocks = StridedBlocks(total);
    DispatchScalar(f64, complex, [&](auto real, auto scalar) {
        using R = decltype(real);
        using S = decltype(scalar);
        ProbeErrorsKernel<R><<<blocks, kThreads, 0, stream>>>(static_cast<S*>(x), n, total);
    });
    return cudaGetLastError();
}

}  // namespace diff_soup_status
