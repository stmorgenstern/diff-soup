// Device scans behind the per-lane status of cuDSS factors held for the phased factor
// contract: one pass over the values of a batch of lanes and over their factor diagonals
// yields, per lane, whether a value or a diagonal entry is non-finite and the smallest and
// largest finite diagonal magnitude (the same scan reads the errors of a batch factor's status
// solve). For a single factor, two kernels keep its status on the
// device without a host wait: one after each numeric phase marks non-finite values, one after
// each solve marks a non-finite solution of finite right-hand sides and writes nan and the
// solve's status. The kernels are compiled by nvcc from status_kernels.cu;
// this header is plain C++ so that the C++ sources can call the launcher.

#ifndef DIFF_SOUP_CUDSS_STATUS_KERNELS_H_
#define DIFF_SOUP_CUDSS_STATUS_KERNELS_H_

#include <cstdint>
#include <cstring>
#include <limits>

#include "cuda_runtime_api.h"

namespace diff_soup_status {

// What one scan found in one lane, as the device writes it. Magnitudes are kept as the bits of
// their float64 value: for non-negative numbers the bit patterns order like the numbers, so the
// device reduces them with integer atomics.
struct LaneStatusWords {
    uint32_t values_nonfinite;     // nonzero when a value of the lane is nan or infinite
    uint32_t diagonal_nonfinite;   // nonzero when a diagonal entry of the lane is nan or infinite
    uint64_t smallest_complement;  // bitwise complement of the bits of the smallest finite
                                   // diagonal magnitude; zero when the lane has none
    uint64_t largest;              // bits of the largest finite diagonal magnitude; zero when none
};

// Queue on `stream` a scan of `lanes` lanes into `out` (device memory, `lanes` entries, zeroed
// by this call before the scan). `values` holds `value_count` entries per lane (lane i at
// i * value_count) and `diagonal` holds `diagonal_count` entries per lane; either may be null
// (or have a zero count) to skip it. Entries are float64 when `f64`, float32 otherwise. Nothing
// is waited for. Returns the first CUDA error of the queued work, cudaSuccess otherwise.
cudaError_t QueueLaneStatusScan(cudaStream_t stream, bool f64, const void* values,
                                int64_t value_count, const void* diagonal,
                                int64_t diagonal_count, int64_t lanes, LaneStatusWords* out);

// The smallest finite diagonal magnitude a scan found; +inf when the lane has none.
inline double SmallestMagnitude(const LaneStatusWords& words) {
    if (words.smallest_complement == 0) return std::numeric_limits<double>::infinity();
    const uint64_t bits = ~words.smallest_complement;
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// The largest finite diagonal magnitude a scan found; zero when the lane has none.
inline double LargestMagnitude(const LaneStatusWords& words) {
    double value = 0.0;
    std::memcpy(&value, &words.largest, sizeof(value));
    return value;
}

// The status solve of a batch factor, whose per-lane diagonals cuDSS does not report: every lane
// solves A_l x = A_l x* for the known solution x*_i = 1 + (i mod 7) / 8 (exact in float32 and
// float64, between 1 and kProbeSolutionLargest), and the error x - x* of each lane, scanned by
// QueueLaneStatusScan, tells whether the batch factor serves that lane.
constexpr double kProbeSolutionLargest = 1.75;

// Queue on `stream` the right-hand sides b_l = A_l x* of `lanes` lanes. A_l is the CSR matrix of
// `n` rows over `offsets` and `columns` (int32, device memory) with lane l's `nse` values at
// `values` + l * nse; b_l is written at `b` + l * n. Entries are float64 when `f64`, float32
// otherwise. Nothing is waited for. Returns the first CUDA error of the queued work.
cudaError_t QueueProbeRightHandSides(cudaStream_t stream, bool f64, const int32_t* offsets,
                                     const int32_t* columns, const void* values, int64_t n,
                                     int64_t nse, int64_t lanes, void* b);

// Queue on `stream` the replacement of every entry of `x` (`lanes` lanes of `n` entries, lane l
// at l * n) by its difference from x*. Nothing is waited for. Returns the first CUDA error of the
// queued work.
cudaError_t QueueProbeErrors(cudaStream_t stream, bool f64, void* x, int64_t n, int64_t lanes);

// Status codes the device writes into a side's verdict words (those of
// diff_soup.linear.SolveStatus).
constexpr int32_t kVerdictSuccess = 0;
constexpr int32_t kVerdictSingular = 1;
constexpr int32_t kVerdictNonfinite = 2;

// The device words of one single factor's status, in device memory, zeroed when the factor is
// created. `status` is the factor's status as the device knows it: NONFINITE when the values of
// its latest numeric phase hold a nan or an infinity, SINGULAR when a solve of finite
// right-hand sides with it came out non-finite (with pivot perturbation off a zero pivot makes
// the solution infinite or nan) or when the host has settled that verdict, SUCCESS otherwise.
// `flags` and `arrived` are the scratch of the kernels below, zero between launches;
// `solve_status` is the status of the latest solve with the factor.
struct SideVerdictWords {
    int32_t status;
    uint32_t flags;
    uint32_t arrived;
    int32_t solve_status;
};

// Up to four int32 words (device addresses) a verdict kernel writes once it has set the status:
// target i takes values[i], except target `status_index` (when it is between 0 and count - 1),
// which takes the status. The handle and status of a one-lane refactor whose verdict is pending,
// written by the kernel that decides the status instead of by a launch of their own.
struct VerdictResult {
    static constexpr int kMax = 4;
    int32_t* targets[kMax] = {nullptr, nullptr, nullptr, nullptr};
    int32_t values[kMax] = {0, 0, 0, 0};
    int count = 0;
    int status_index = -1;
};

// Queue on `stream` one kernel over the `count` values of a factor's latest numeric phase
// (float64 when `f64`, float32 otherwise) that sets `words->status` to NONFINITE when one of
// them is nan or infinite and to SUCCESS otherwise, then writes `result` when it is given.
// Nothing is waited for. Returns the first CUDA error of the queued work.
cudaError_t QueueValuesVerdict(cudaStream_t stream, bool f64, const void* values, int64_t count,
                               SideVerdictWords* words, const VerdictResult* result = nullptr);

// Queue on `stream` one kernel after a solve with a factor whose words are `words`: `total`
// right-hand-side entries at `b` and solution entries at `x` (float64 when `f64`). When the
// factor's status is SUCCESS, every right-hand-side entry is finite and a solution entry is not,
// the factor's status becomes SINGULAR. When the factor's status is then other than SUCCESS,
// every solution entry is set to nan. The solve's status is written to `words->solve_status`
// and, when `out` is not null, to `out` (device). Nothing is waited for. Returns the first CUDA
// error of the queued work.
cudaError_t QueueSolveVerdict(cudaStream_t stream, bool f64, const void* b, void* x,
                              int64_t total, SideVerdictWords* words, int32_t* out);

}  // namespace diff_soup_status

#endif  // DIFF_SOUP_CUDSS_STATUS_KERNELS_H_
