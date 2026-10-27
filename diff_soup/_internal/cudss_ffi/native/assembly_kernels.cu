// The device assembly kernel of the cuDSS factor contract (see assembly_kernels.h): one thread
// per CSR position and lane gathers the coordinate entries that add into that position, so the
// sum is deterministic (no atomics), the writes are coalesced in CSR order and duplicate
// coordinates add in the order the analysis listed them. An entry in a Dirichlet row is
// multiplied by zero rather than skipped, so a non-finite value there still reaches the sum and
// the factorization reports it, as every backend's assembly does.

#include "assembly_kernels.h"

namespace diff_soup_assembly {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 2147483647;

template <typename In, typename Out>
__global__ void __launch_bounds__(kThreads)
    AssembleKernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ entries,
                   const uint8_t* __restrict__ diagonal_add, int64_t nnz, int64_t nse,
                   int64_t total, const In* __restrict__ in, Out* __restrict__ out) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total;
         i += stride) {
        const int64_t lane = i / nse;
        const int64_t position = i - lane * nse;
        const In* values = in + lane * nnz;
        In sum = In(0);
        const int32_t begin = offsets[position];
        const int32_t end = offsets[position + 1];
        for (int32_t k = begin; k < end; ++k) {
            const int32_t entry = entries[k];
            // A Dirichlet-row entry contributes value * 0: nan or inf stays non-finite.
            sum += entry >= 0 ? values[entry] : values[-entry - 1] * In(0);
        }
        if (diagonal_add != nullptr && diagonal_add[position] != 0) sum += In(1);
        out[i] = static_cast<Out>(sum);
    }
}

template <typename In, typename Out>
cudaError_t Launch(cudaStream_t stream, const AssemblyMap& map, const void* in, void* out,
                   int64_t lanes) {
    const int64_t total = lanes * map.nse;
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    AssembleKernel<In, Out><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        map.offsets, map.entries, map.diagonal_add, map.nnz, map.nse, total,
        static_cast<const In*>(in), static_cast<Out*>(out));
    return cudaGetLastError();
}

// One thread per CSR position and lane, as AssembleKernel, writing only the entries that differ
// from the ones `out` holds and marking their lane, and copying the lane's source entries into
// `kept` when it is given (see QueueAssemblyCompare). Every coordinate entry belongs to exactly
// one CSR position, so each `kept` entry is written by one thread. `kept` may be `in` itself and
// is therefore not declared restrict; a lane writes it only where it reads `held` (or where
// `kept` is another buffer), never an entry it reads through `in`.
template <typename In, typename Out>
__global__ void __launch_bounds__(kThreads)
    AssembleCompareKernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ entries,
                          const uint8_t* __restrict__ diagonal_add, int64_t nnz, int64_t nse,
                          int64_t total, const In* __restrict__ in, const In* __restrict__ held,
                          const uint8_t* __restrict__ predicate,
                          const int32_t* __restrict__ held_slot, Out* __restrict__ out,
                          int32_t* __restrict__ fixed, int32_t* __restrict__ changed,
                          int32_t tag, In* kept, ResultWords words) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total;
         i += stride) {
        const int64_t lane = i / nse;
        const int64_t position = i - lane * nse;
        const bool asks = predicate != nullptr && predicate[lane] != 0;
        const bool empty =
            predicate != nullptr && !asks && held_slot != nullptr && held_slot[lane] < 0;
        // One thread of a lane writes the lane's fixed bits into `fixed`; the threads that find a
        // changed entry all write the same word into `changed`, so no two writes conflict.
        if (position == 0) {
            if (asks) {
                fixed[lane] = (tag << 3) | kLaneAsks;
            } else if (empty) {
                fixed[lane] = (tag << 3) | kLaneEmpty;
            }
            if (lane == 0 && !empty) {
                for (int w = 0; w < words.count; ++w) {
                    *words.targets[w] =
                        words.sources[w] != nullptr ? *words.sources[w] : words.values[w];
                }
            }
        }
        const In* values = (asks || held == nullptr ? in : held) + lane * nnz;
        In* lane_kept = kept != nullptr && (!asks || kept != in) ? kept + lane * nnz : nullptr;
        const int32_t begin = offsets[position];
        const int32_t end = offsets[position + 1];
        if (empty) {
            if (lane_kept != nullptr) {
                for (int32_t k = begin; k < end; ++k) {
                    const int32_t entry = entries[k] >= 0 ? entries[k] : -entries[k] - 1;
                    lane_kept[entry] = values[entry];
                }
            }
            continue;
        }
        In sum = In(0);
        for (int32_t k = begin; k < end; ++k) {
            const int32_t entry = entries[k];
            const int32_t index = entry >= 0 ? entry : -entry - 1;
            const In value = values[index];
            if (lane_kept != nullptr) lane_kept[index] = value;
            sum += entry >= 0 ? value : value * In(0);
        }
        if (diagonal_add != nullptr && diagonal_add[position] != 0) sum += In(1);
        const Out assembled = static_cast<Out>(sum);
        // A nan never equals the entry it replaces, so a non-finite lane always counts as
        // changed and is factored (and reported) again.
        if (!(assembled == out[i])) {
            out[i] = assembled;
            changed[lane] = (tag << 3) | kLaneChanged;
        }
    }
}

template <typename In, typename Out>
cudaError_t LaunchCompare(cudaStream_t stream, const AssemblyMap& map, const void* in,
                          const void* held, const uint8_t* predicate, const int32_t* held_slot,
                          void* out, int64_t lanes, int32_t* fixed, int32_t* changed,
                          int32_t tag, void* kept, const ResultWords& words) {
    const int64_t total = lanes * map.nse;
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    AssembleCompareKernel<In, Out><<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        map.offsets, map.entries, map.diagonal_add, map.nnz, map.nse, total,
        static_cast<const In*>(in), static_cast<const In*>(held), predicate, held_slot,
        static_cast<Out*>(out), fixed, changed, tag, static_cast<In*>(kept), words);
    return cudaGetLastError();
}

template <typename In>
cudaError_t LaunchCompareOut(cudaStream_t stream, const AssemblyMap& map, const void* in,
                             const void* held, const uint8_t* predicate,
                             const int32_t* held_slot, Scalar out_type, void* out, int64_t lanes,
                             int32_t* fixed, int32_t* changed, int32_t tag, void* kept,
                             const ResultWords& words) {
    if (out_type == Scalar::kFloat64) {
        return LaunchCompare<In, double>(stream, map, in, held, predicate, held_slot, out, lanes,
                                         fixed, changed, tag, kept, words);
    }
    if (out_type == Scalar::kFloat32) {
        return LaunchCompare<In, float>(stream, map, in, held, predicate, held_slot, out, lanes,
                                        fixed, changed, tag, kept, words);
    }
    return cudaErrorInvalidValue;
}

template <typename In>
cudaError_t LaunchOut(cudaStream_t stream, const AssemblyMap& map, const void* in,
                      Scalar out_type, void* out, int64_t lanes) {
    if (out_type == Scalar::kFloat64) return Launch<In, double>(stream, map, in, out, lanes);
    if (out_type == Scalar::kFloat32) return Launch<In, float>(stream, map, in, out, lanes);
    return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t QueueAssembly(cudaStream_t stream, const AssemblyMap& map, Scalar in_type,
                          const void* in, Scalar out_type, void* out, int64_t lanes) {
    if (lanes <= 0 || map.nse <= 0) return cudaSuccess;
    if (map.offsets == nullptr || map.entries == nullptr || in == nullptr || out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (in_type == Scalar::kFloat64) {
        return LaunchOut<double>(stream, map, in, out_type, out, lanes);
    }
    if (in_type == Scalar::kFloat32) return LaunchOut<float>(stream, map, in, out_type, out, lanes);
    return cudaErrorInvalidValue;
}

cudaError_t QueueAssemblyCompare(cudaStream_t stream, const AssemblyMap& map, Scalar in_type,
                                 const void* in, const void* held, const uint8_t* predicate,
                                 const int32_t* held_slot, Scalar out_type, void* out,
                                 int64_t lanes, int32_t* fixed, int32_t* changed, int32_t tag,
                                 void* kept, const ResultWords* words) {
    if (lanes <= 0 || map.nse <= 0) return cudaSuccess;
    if (fixed == nullptr || changed == nullptr || tag <= 0 || tag >= (1 << 28) || map.offsets == nullptr ||
        map.entries == nullptr || in == nullptr || out == nullptr ||
        (predicate != nullptr && held == nullptr) ||
        (words != nullptr && (words->count < 0 || words->count > ResultWords::kMax))) {
        return cudaErrorInvalidValue;
    }
    const ResultWords none;
    const ResultWords& write = words != nullptr ? *words : none;
    if (in_type == Scalar::kFloat64) {
        return LaunchCompareOut<double>(stream, map, in, held, predicate, held_slot, out_type, out,
                                        lanes, fixed, changed, tag, kept, write);
    }
    if (in_type == Scalar::kFloat32) {
        return LaunchCompareOut<float>(stream, map, in, held, predicate, held_slot, out_type, out,
                                       lanes, fixed, changed, tag, kept, write);
    }
    return cudaErrorInvalidValue;
}

namespace {

__global__ void ResultWordsKernel(ResultWords words) {
    const int i = static_cast<int>(threadIdx.x);
    if (i < words.count) {
        *words.targets[i] = words.sources[i] != nullptr ? *words.sources[i] : words.values[i];
    }
}

// Fill `count` 32-bit words at `out` with `pattern` (and the `tail` bytes after them with its low
// byte); the first thread also writes `words`.
__global__ void __launch_bounds__(kThreads)
    FillAndWordsKernel(uint32_t* __restrict__ out, int64_t count, uint32_t pattern, int tail,
                       ResultWords words) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    for (int64_t i = first; i < count; i += stride) out[i] = pattern;
    if (first == 0) {
        auto* bytes = reinterpret_cast<uint8_t*>(out + count);
        for (int t = 0; t < tail; ++t) bytes[t] = static_cast<uint8_t>(pattern);
        for (int w = 0; w < words.count; ++w) {
            *words.targets[w] = words.sources[w] != nullptr ? *words.sources[w] : words.values[w];
        }
    }
}

}  // namespace

cudaError_t QueueResultWords(cudaStream_t stream, const ResultWords& words) {
    if (words.count <= 0) return cudaSuccess;
    if (words.count > ResultWords::kMax) return cudaErrorInvalidValue;
    ResultWordsKernel<<<1, ResultWords::kMax, 0, stream>>>(words);
    return cudaGetLastError();
}

cudaError_t QueueFillAndWords(cudaStream_t stream, void* out, size_t bytes, uint8_t byte,
                              const ResultWords& words) {
    if (words.count < 0 || words.count > ResultWords::kMax) return cudaErrorInvalidValue;
    if (bytes > 0 && out == nullptr) return cudaErrorInvalidValue;
    // A device allocation is aligned far beyond four bytes, and the solutions the handlers fill
    // start at whole entries of float32 or float64, so the words below are aligned.
    if (reinterpret_cast<uintptr_t>(out) % sizeof(uint32_t) != 0) return cudaErrorInvalidValue;
    const int64_t count = static_cast<int64_t>(bytes / sizeof(uint32_t));
    const int tail = static_cast<int>(bytes % sizeof(uint32_t));
    const uint32_t pattern = 0x01010101u * static_cast<uint32_t>(byte);
    int64_t blocks = (count + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;
    FillAndWordsKernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        static_cast<uint32_t*>(out), count, pattern, tail, words);
    return cudaGetLastError();
}

namespace {

__global__ void __launch_bounds__(kThreads)
    StackedPatternKernel(const int32_t* __restrict__ offsets,
                         const int32_t* __restrict__ columns, int64_t n, int64_t nse,
                         int64_t lanes, int32_t* __restrict__ out_offsets,
                         int32_t* __restrict__ out_columns) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    const int64_t rows = lanes * n;
    const int64_t entries = lanes * nse;
    const int64_t total = rows + 1 > entries ? rows + 1 : entries;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < total;
         i += stride) {
        if (i < rows) {
            const int64_t lane = i / n;
            out_offsets[i] = static_cast<int32_t>(lane * nse + offsets[i - lane * n]);
        } else if (i == rows) {
            out_offsets[i] = static_cast<int32_t>(entries);
        }
        if (i < entries) {
            const int64_t lane = i / nse;
            out_columns[i] = static_cast<int32_t>(lane * n + columns[i - lane * nse]);
        }
    }
}

}  // namespace

cudaError_t QueueStackedPattern(cudaStream_t stream, const int32_t* offsets,
                                const int32_t* columns, int64_t n, int64_t nse, int64_t lanes,
                                int32_t* out_offsets, int32_t* out_columns) {
    const int64_t rows = lanes * n;
    const int64_t entries = lanes * nse;
    const int64_t total = rows + 1 > entries ? rows + 1 : entries;
    int64_t blocks = (total + kThreads - 1) / kThreads;
    if (blocks > 65535) blocks = 65535;
    StackedPatternKernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        offsets, columns, n, nse, lanes, out_offsets, out_columns);
    return cudaGetLastError();
}

}  // namespace diff_soup_assembly
