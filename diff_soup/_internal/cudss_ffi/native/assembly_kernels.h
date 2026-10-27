// Device assembly of a sparse operator's values for the cuDSS factor contract: the values a
// caller passes in the coordinate order of its topology (one entry per coordinate, duplicates
// allowed, every lane of a batch after the other) become the CSR values the factorization reads,
// in one kernel per call from a map built once per analysis. The kernel is compiled by nvcc
// from assembly_kernels.cu; this header is plain C++ so that the C++ sources can call the
// launcher.

#ifndef DIFF_SOUP_CUDSS_ASSEMBLY_KERNELS_H_
#define DIFF_SOUP_CUDSS_ASSEMBLY_KERNELS_H_

#include <cstdint>

#include "cuda_runtime_api.h"

namespace diff_soup_assembly {

// Scalar types the assembly reads and writes.
enum class Scalar : int { kFloat64 = 0, kFloat32 = 1 };

// The device-resident map of one CSR pattern (one orientation of an analysis):
//   offsets[p] .. offsets[p + 1]  the positions in `entries` of the coordinate entries that add
//                                 into CSR position p (offsets has nse + 1 entries; every
//                                 coordinate entry appears once in `entries`);
//   entries[k]                    a coordinate entry index e in [0, nnz), or -(e + 1) for an
//                                 entry that lies in a Dirichlet row: its value is multiplied by
//                                 zero (so a non-finite value stays non-finite) before it adds;
//   diagonal_add[p]               nonzero where CSR position p is the diagonal of a Dirichlet
//                                 row, which receives one; null when the pattern has none.
struct AssemblyMap {
    const int32_t* offsets = nullptr;
    const int32_t* entries = nullptr;
    const uint8_t* diagonal_add = nullptr;
    int64_t nnz = 0;  // coordinate entries per lane
    int64_t nse = 0;  // CSR positions per lane
};

// Queue on `stream` the assembly of `lanes` lanes: lane i reads `nnz` entries of `in_type` at
// `in` + i * nnz and writes `nse` entries of `out_type` at `out` + i * nse, each CSR position
// the sum of its coordinate entries (in `in_type`) plus one on a Dirichlet diagonal, cast to
// `out_type`. Nothing is waited for. Returns the first CUDA error of the queued work.
cudaError_t QueueAssembly(cudaStream_t stream, const AssemblyMap& map, Scalar in_type,
                          const void* in, Scalar out_type, void* out, int64_t lanes);

// Up to four int32 words, each written to its own device address, by one kernel whose arguments
// carry the values: the handle and status of a one-lane call, written without a host-to-device
// copy per array. A word whose `sources` entry is not null takes the device word it points at
// (read when the kernel runs) instead of its value: a status the device holds.
struct ResultWords {
    static constexpr int kMax = 4;
    int32_t* targets[kMax] = {nullptr, nullptr, nullptr, nullptr};
    int32_t values[kMax] = {0, 0, 0, 0};
    const int32_t* sources[kMax] = {nullptr, nullptr, nullptr, nullptr};
    int count = 0;
};

// Bits of the per-lane words QueueAssemblyCompare writes.
constexpr int32_t kLaneChanged = 1;  // an assembled entry of the lane differed from the copy's
constexpr int32_t kLaneAsks = 2;     // the lane's predicate holds
constexpr int32_t kLaneEmpty = 4;    // the lane does not ask and its held handle is empty

// Queue on `stream` the assembly of `lanes` lanes into `out` in place, comparing every assembled
// entry with the entry it replaces, and mark the lanes in two int32 words each: `fixed` (lane i
// at fixed[i]: kLaneAsks or kLaneEmpty) and `changed` (kLaneChanged). A word this call writes is
// `tag << 3` plus its bit; a word nothing marks keeps what an earlier call wrote, so the caller
// reads a word whose `>> 3` differs from `tag` (a fresh tag per call, 0 < tag < 2^28) as 0, and
// no clearing of the words is queued. A lane is kLaneChanged when any assembled entry differs
// from the one `out` held (a non-finite entry always differs). Lane i assembles:
//   * without `predicate`, `in` + i * nnz;
//   * with `predicate` (one byte per lane): `in` + i * nnz where the predicate holds (kLaneAsks);
//     nothing where it is false and the `held_slot` entry is negative (an empty handle;
//     `held_slot` may be null), which is marked kLaneEmpty and left alone; `held` + i * nnz (the
//     values its held factor carries) otherwise.
// Entries equal to the ones `out` holds are not written. `predicate` and `held_slot` are device
// arrays of `lanes` entries; `held` has the layout and type of `in`.
//
// With `kept` (the layout and type of `in`; it may be `in` itself), the same pass writes each
// lane's source values into `kept` + i * nnz: `held` + i * nnz for a lane whose predicate is
// false (an empty lane included), `in` + i * nnz for one whose predicate holds (nothing moves
// when `kept` is `in`). With `words` (count above zero, for a one-lane call), the thread of lane
// 0 and CSR position 0 also writes those words as QueueResultWords does, unless lane 0 is
// kLaneEmpty; nothing writes them when the pattern has no CSR position. Nothing is waited for.
// Returns the first CUDA error of the queued work.
cudaError_t QueueAssemblyCompare(cudaStream_t stream, const AssemblyMap& map, Scalar in_type,
                                 const void* in, const void* held, const uint8_t* predicate,
                                 const int32_t* held_slot, Scalar out_type, void* out,
                                 int64_t lanes, int32_t* fixed, int32_t* changed, int32_t tag,
                                 void* kept = nullptr, const ResultWords* words = nullptr);

// Queue on `stream` the writes of `words`. Nothing is waited for. Returns the first CUDA error of
// the queued work.
cudaError_t QueueResultWords(cudaStream_t stream, const ResultWords& words);

// Queue on `stream` one kernel that sets every one of the `bytes` bytes at `out` to `byte` (zero
// for a cleared solution, 0xFF for a nan solution in float32 and float64) and writes `words` as
// QueueResultWords does: the status of a one-lane solve, known before the solve runs, written
// with the fill of its solution instead of by a launch of its own. Nothing is waited for. Returns
// the first CUDA error of the queued work.
cudaError_t QueueFillAndWords(cudaStream_t stream, void* out, size_t bytes, uint8_t byte,
                              const ResultWords& words);

// Queue on `stream` the CSR pattern of `lanes` copies of one n-row pattern (`offsets`, n + 1
// entries; `columns`, nse entries) stacked block-diagonally into one matrix of lanes * n rows:
// `out_offsets` (lanes * n + 1 entries) and `out_columns` (lanes * nse entries), copy l shifted
// by l * nse positions and l * n columns. Every count must fit int32. Nothing is waited for.
// Returns the first CUDA error of the queued work.
cudaError_t QueueStackedPattern(cudaStream_t stream, const int32_t* offsets,
                                const int32_t* columns, int64_t n, int64_t nse, int64_t lanes,
                                int32_t* out_offsets, int32_t* out_columns);

}  // namespace diff_soup_assembly

#endif  // DIFF_SOUP_CUDSS_ASSEMBLY_KERNELS_H_
