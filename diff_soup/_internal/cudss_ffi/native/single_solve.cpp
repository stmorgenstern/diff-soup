#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <limits>
#include <string>

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"

// The extension is written against the cuDSS 1.x API (CUDA's cudaDataType_t in the matrix
// descriptors, the 1.x configuration and data parameters) and links libcudss.so.1. The release
// of the headers it compiles against is recorded below and reported by cudss_built_version().
#if defined(CUDSS_VERSION_MAJOR)
#define DIFF_SOUP_CUDSS_MAJOR CUDSS_VERSION_MAJOR
#define DIFF_SOUP_CUDSS_MINOR CUDSS_VERSION_MINOR
#define DIFF_SOUP_CUDSS_PATCH CUDSS_VERSION_PATCH
#elif defined(CUDSS_VER_MAJOR)
#define DIFF_SOUP_CUDSS_MAJOR CUDSS_VER_MAJOR
#define DIFF_SOUP_CUDSS_MINOR CUDSS_VER_MINOR
#define DIFF_SOUP_CUDSS_PATCH CUDSS_VER_PATCH
#elif defined(CUDSS_VERSION)
#define DIFF_SOUP_CUDSS_MAJOR (CUDSS_VERSION / 10000)
#define DIFF_SOUP_CUDSS_MINOR ((CUDSS_VERSION / 100) % 100)
#define DIFF_SOUP_CUDSS_PATCH (CUDSS_VERSION % 100)
#else
#error "cudss.h defines no version macro; build against the cuDSS 1.x headers of nvidia-cudss-cu12 or nvidia-cudss-cu13."
#endif
#if DIFF_SOUP_CUDSS_MAJOR < 1
#error "cuDSS 1.0 or newer is required; the cudss.h on the include path is older. Build against nvidia-cudss-cu12 or nvidia-cudss-cu13 1.x."
#endif
// CUDSS_VERSION encodes major * 10000 + minor * 100 + patch.
#if defined(CUDSS_VERSION) && CUDSS_VERSION < 10000
#error "cuDSS 1.0 or newer is required; the cudss.h on the include path reports CUDSS_VERSION below 10000."
#endif

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {
using Clock = std::chrono::steady_clock;

// Host-side sparsity patterns, registered once per content-stable instance id
// from Python before the first solve. A handler whose pattern attributes are
// empty reads its pattern here, so the program text carries only the id and a
// compiled program's size and cache key do not grow with the mesh.
struct HostTopology {
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
};

std::mutex& TopologyMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<int64_t, std::shared_ptr<const HostTopology>>& TopologyTable() {
    static std::unordered_map<int64_t, std::shared_ptr<const HostTopology>> table;
    return table;
}

void RegisterTopology(int64_t instance_id, const int32_t* offsets, size_t n_offsets,
                      const int32_t* columns, size_t n_columns) {
    auto topology = std::make_shared<HostTopology>();
    topology->offsets.assign(offsets, offsets + n_offsets);
    topology->columns.assign(columns, columns + n_columns);
    std::lock_guard<std::mutex> lock(TopologyMutex());
    TopologyTable()[instance_id] = std::move(topology);
}

// Resolves the pattern a handler works with: the attributes when they carry
// one, otherwise the registered pattern. `keep` owns the registered storage
// for the duration of the call.
ffi::Error ResolveTopology(int64_t instance_id,
                           ffi::Span<const int32_t>& csr_offsets,
                           ffi::Span<const int32_t>& csr_columns,
                           std::shared_ptr<const HostTopology>& keep) {
    if (csr_offsets.size() > 0) {
        return ffi::Error::Success();
    }
    {
        std::lock_guard<std::mutex> lock(TopologyMutex());
        auto it = TopologyTable().find(instance_id);
        if (it != TopologyTable().end()) {
            keep = it->second;
        }
    }
    if (!keep) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "cuDSS instance " + std::to_string(instance_id) +
                " has no sparsity pattern: the call carries empty csr_offsets and "
                "csr_columns attributes and none was registered for this id. Register "
                "the pattern (register_topology) in this process before the first "
                "solve, or pass the pattern attributes.");
    }
    csr_offsets = ffi::Span<const int32_t>(keep->offsets.data(), keep->offsets.size());
    csr_columns = ffi::Span<const int32_t>(keep->columns.data(), keep->columns.size());
    return ffi::Error::Success();
}

struct SolverDebugStats {
    int64_t execute_calls = 0;
    int64_t handler_calls = 0;
    int64_t analysis_calls = 0;
    int64_t factorization_calls = 0;
    int64_t refactorization_calls = 0;
    int64_t solve_phase_calls = 0;
    int64_t cold_calls = 0;
    int64_t warm_calls = 0;
    int64_t refit_calls = 0;
    int64_t solve_calls = 0;
    // The ``*_ms`` fields are steady_clock spans measured on the HOST around
    // the cuDSS API calls: enqueue/dispatch time, NOT device execution time
    // (no synchronization is inserted). They surface as ``*_host_ms`` keys.
    // The ``*_cuda_ms`` fields are CUDA-event device spans where recorded.
    double total_ms = 0.0;
    double set_stream_ms = 0.0;
    double set_csr_ms = 0.0;
    double set_values_ms = 0.0;
    double analysis_ms = 0.0;
    double factorization_ms = 0.0;
    double refactorization_ms = 0.0;
    double solve_phase_ms = 0.0;
    double analysis_cuda_ms = 0.0;
    double factorization_cuda_ms = 0.0;
    double refactorization_cuda_ms = 0.0;
    double solve_phase_cuda_ms = 0.0;
    // The queried CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN device
    // floor (bytes) from the last cold-start analysis under hybrid mode; -1 when
    // hybrid mode is off or the query is unavailable. NOT accumulated — a
    // last-seen value (set, not added), surfaced to the host LinearState.
    int64_t hybrid_device_memory_min = -1;
    // CUDSS_DATA_MEMORY_ESTIMATES queried after the
    // cold-start ANALYSIS (single path). Slots 0-5 of the 16xint64 layout:
    // permanent device / peak device / permanent host / peak host /
    // hybrid min device / hybrid max host (bytes). -1 = not queried. Last-seen
    // semantics like hybrid_device_memory_min.
    int64_t memory_estimates[6] = {-1, -1, -1, -1, -1, -1};
    // Facts about the factor built by the last cold factorization (last-seen):
    // nnz(L + U), the factorization flop count, the superpanel count; -1 = not
    // queried.
    int64_t lu_nnz = -1;
    int64_t factor_flops = -1;
    int64_t nsuperpanels = -1;
};

std::mutex g_stats_mu;
std::unordered_map<int64_t, SolverDebugStats> g_stats;

inline double ElapsedMs(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void AccumulateStats(int64_t instance_id, const SolverDebugStats& delta) {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    SolverDebugStats& out = g_stats[instance_id];
    out.execute_calls += delta.execute_calls;
    out.handler_calls += delta.handler_calls;
    out.analysis_calls += delta.analysis_calls;
    out.factorization_calls += delta.factorization_calls;
    out.refactorization_calls += delta.refactorization_calls;
    out.solve_phase_calls += delta.solve_phase_calls;
    out.cold_calls += delta.cold_calls;
    out.warm_calls += delta.warm_calls;
    out.refit_calls += delta.refit_calls;
    out.solve_calls += delta.solve_calls;
    out.total_ms += delta.total_ms;
    out.set_stream_ms += delta.set_stream_ms;
    out.set_csr_ms += delta.set_csr_ms;
    out.set_values_ms += delta.set_values_ms;
    out.analysis_ms += delta.analysis_ms;
    out.factorization_ms += delta.factorization_ms;
    out.refactorization_ms += delta.refactorization_ms;
    out.solve_phase_ms += delta.solve_phase_ms;
    out.analysis_cuda_ms += delta.analysis_cuda_ms;
    out.factorization_cuda_ms += delta.factorization_cuda_ms;
    out.refactorization_cuda_ms += delta.refactorization_cuda_ms;
    out.solve_phase_cuda_ms += delta.solve_phase_cuda_ms;
    // Last-seen, not accumulated: a cold start under hybrid mode overwrites it;
    // warm calls leave it untouched (delta carries -1).
    if (delta.hybrid_device_memory_min >= 0) {
        out.hybrid_device_memory_min = delta.hybrid_device_memory_min;
    }
    // Last-seen: peak-device slot >= 0 marks a fresh query.
    if (delta.memory_estimates[1] >= 0) {
        for (int i = 0; i < 6; ++i) out.memory_estimates[i] = delta.memory_estimates[i];
    }
    if (delta.lu_nnz >= 0) out.lu_nnz = delta.lu_nnz;
    if (delta.factor_flops >= 0) out.factor_flops = delta.factor_flops;
    if (delta.nsuperpanels >= 0) out.nsuperpanels = delta.nsuperpanels;
}

SolverDebugStats GetStatsSnapshot(int64_t instance_id) {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    auto it = g_stats.find(instance_id);
    if (it == g_stats.end()) {
        return SolverDebugStats{};
    }
    return it->second;
}

void ResetStats(int64_t instance_id) {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    g_stats[instance_id] = SolverDebugStats{};
}

int WrappedDeviceAlloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
    (void)ctx;
    cudaError_t err = cudaMallocAsync(ptr, size, stream);
    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        err = cudaMalloc(ptr, size);
    }
    return static_cast<int>(err);
}

int WrappedDeviceFree(void* ctx, void* ptr, size_t size, cudaStream_t stream) {
    (void)ctx;
    (void)size;
    cudaError_t err = cudaFreeAsync(ptr, stream);
    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        err = cudaFree(ptr);
    }
    return static_cast<int>(err);
}

bool DebugPtrsEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DIFF_SOUP_CUDSS_DEBUG_PTRS");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

bool PhaseCudaTimingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DIFF_SOUP_CUDSS_CUDA_EVENTS");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

int DebugPtrsMaxCalls() {
    static const int max_calls = []() {
        const char* env = std::getenv("DIFF_SOUP_CUDSS_DEBUG_MAX_CALLS");
        if (env == nullptr || env[0] == '\0') {
            return 64;
        }
        int parsed = std::atoi(env);
        return parsed > 0 ? parsed : 64;
    }();
    return max_calls;
}

bool ShouldTrace(int64_t call_count) {
    return DebugPtrsEnabled() && call_count < static_cast<int64_t>(DebugPtrsMaxCalls());
}

void PrintMemInfo(const char* tag) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        printf(
            "[cudss_debug] %s mem free=%zu total=%zu used=%zu\n",
            tag,
            free_bytes,
            total_bytes,
            total_bytes - free_bytes
        );
    } else {
        printf("[cudss_debug] %s cudaMemGetInfo error=%d\n", tag, static_cast<int>(err));
    }
}

}  // namespace

// verification ================================================================
// The CUDA error pending on the calling thread, consumed, as a sentence to append to a cuDSS
// failure; empty when none is pending. cuDSS reports a CUDA error that was pending when its call
// began, or that its call raised, through its own status, and the CUDA error's name says which
// failure it was. VERIFY: when cuDSS consumes the error itself, none is pending any more here.
static std::string PendingCudaError() {
    const cudaError_t pending = cudaGetLastError();
    if (pending == cudaSuccess) return std::string();
    return std::string(" CUDA error pending: ") + cudaGetErrorName(pending) + " (" +
           cudaGetErrorString(pending) + ").";
}

// A failed cuDSS call. Out of memory (CUDSS_STATUS_ALLOC_FAILED) is reported
// as a resource-exhausted error that names every visible device's free and
// total memory, so the caller can size the fix (more devices on the model
// axis, host spill, a reduced solve mode); every other status stays an
// internal error carrying the phase that failed and its status. Either names the
// CUDA error pending on the thread, when there is one.
static ffi::Error CudssCallFailed(cudssStatus_t status, const char* phase,
                                  int64_t peak_device_estimate = -1) {
    const std::string cuda = PendingCudaError();
    if (status != CUDSS_STATUS_ALLOC_FAILED) {
        return ffi::Error::Internal(std::string("cuDSS call failed: ") + phase +
                                    " returned status " +
                                    std::to_string(static_cast<int>(status)) + "." + cuda);
    }
    std::string text = "cuDSS ran out of memory (CUDSS_STATUS_ALLOC_FAILED) in ";
    text += phase;
    text += ".";
    text += cuda;
    if (peak_device_estimate > 0) {
        char line[96];
        snprintf(line, sizeof(line), " cuDSS estimated %.2f GB of peak device memory for this factorization.",
                 peak_device_estimate / 1e9);
        text += line;
    }
    int current = -1;
    int count = 0;
    if (cudaGetDevice(&current) == cudaSuccess && cudaGetDeviceCount(&count) == cudaSuccess) {
        for (int d = 0; d < count; ++d) {
            size_t free_b = 0;
            size_t total_b = 0;
            if (cudaSetDevice(d) == cudaSuccess && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                char line[128];
                snprintf(line, sizeof(line), " Device %d has %.2f GB free of %.2f GB.",
                         d, free_b / 1e9, total_b / 1e9);
                text += line;
            }
        }
        cudaSetDevice(current);
        (void)cudaGetLastError();
    }
    text += " The factorization needs more device memory than is free: spread it over "
            "more devices on the layout's model axis, enable Layout.hybrid_memory with a "
            "device memory limit, or use a reduced solve_mode.";
    return ffi::Error(ffi::ErrorCode::kResourceExhausted, text);
}

#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("[diff_soup cudss] FAILED: cuDSS call returned status = %d, details: " #msg "\n", status); \
            return CudssCallFailed(status, #msg); \
        } \
    } while(0);

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err)); \
      return ffi::Error::Internal("A CUDA call failed."); \
    } \
  } while (0)

// cuDSS (0.7.1 and 0.8.0) warm SOLVE phases fold the output buffer's
// pre-call contents into the result above roughly 1e5 dofs: NaN garbage in x
// becomes an all-NaN solution, finite garbage a silently wrong tail suffix,
// with CUDSS_STATUS_SUCCESS throughout. Under XLA the output is recycled
// pool memory, so garbage is the common case. Zero-filling x together with
// at least one iterative-refinement step (the Python layer's ir_nsteps
// default) restores exact results in every reproduced failing lane; each
// half alone is insufficient. The zero-fill must land before the call's
// FIRST cuDSS phase, not merely before SOLVE — a refactorization executed
// with garbage in the registered x descriptor already poisons the
// subsequent solve. It rides the caller's stream, which orders it ahead of
// single-device cuDSS work; under MG the caller's stream is drained before
// any phase can read x.
static ffi::Error ZeroSolveOutput(
    void* out, size_t bytes, cudaStream_t stream, bool mg_active
) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, bytes, stream));
    if (mg_active) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return ffi::Error::Success();
}

// Whether the per-call solver zeroes its solution buffer before a call. cuDSS 1.0 wrote no stale
// output into 10 warm float64 solves without the fill and without refinement, at n = 250,000 and
// n = 1,000,000 on an L40S, so float64 solves skip it; other scalar types, not measured, keep it.
template <ffi::DataType T>
constexpr bool ZeroFillsSolveOutput() {
    return T != ffi::F64;
}

// MG solves read buffers on peer devices, but stream-ordered
// (cudaMallocAsync) pool allocations — XLA's allocation style — are not
// peer-accessible until peer access is enabled between the devices; without
// it the k=2 solve fails at the same size threshold even outside XLA.
// Enable access across every ordered pair of the MG device set before the
// MG handle is created. Enablement is per device pair per process and
// sticky, so a repeat call returns cudaErrorPeerAccessAlreadyEnabled —
// success here. Devices without a peer link (can == 0) are left to cuDSS,
// which routes those transfers through host staging.
static ffi::Error EnableMgPeerAccess(
    const std::vector<int>& mg_indices, int restore_device
) {
    for (size_t i = 0; i < mg_indices.size(); ++i) {
        CUDA_CHECK(cudaSetDevice(mg_indices[i]));
        for (size_t j = 0; j < mg_indices.size(); ++j) {
            if (i == j) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, mg_indices[i], mg_indices[j]));
            if (!can) continue;
            cudaError_t err = cudaDeviceEnablePeerAccess(mg_indices[j], 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();  // absorb the sticky error state
                continue;
            }
            if (err != cudaSuccess) {
                printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,
                       cudaGetErrorString(err));
                return ffi::Error::Internal(
                    "cudaDeviceEnablePeerAccess failed for the MG device set.");
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(restore_device));
    return ffi::Error::Success();
}

// The CUDA ordinals cuDSS MG spreads a factor over for a call on `device`: the model group of
// `all` (consecutive groups of `group_size` ordinals, as the analysis lists the model groups of
// this process) that holds `device`, rotated to start at it. Returns the refusal, and leaves
// `group` empty, when `all` is not a whole number of groups, names a device this process does not
// see or one device twice, or holds no group with `device`; an empty string otherwise.
static std::string ModelGroupOf(const std::vector<int>& all, int64_t group_size, int device,
                                std::vector<int>* group) {
    group->clear();
    const size_t size = static_cast<size_t>(std::max<int64_t>(group_size, 1));
    std::string listed;
    for (size_t i = 0; i < all.size(); ++i) {
        listed += (i % size == 0 ? (i ? "], [" : "[") : ", ") + std::to_string(all[i]);
    }
    listed += "]";
    if (group_size < 2 || all.size() % size != 0) {
        return "cuDSS MG mode: mg_devices holds " + std::to_string(all.size()) +
               " CUDA ordinals, not a whole number of model groups of " +
               std::to_string(group_size) + " devices.";
    }
    int ndev = 0;
    const cudaError_t counted = cudaGetDeviceCount(&ndev);
    if (counted != cudaSuccess) {
        return std::string("cuDSS MG mode: cudaGetDeviceCount failed: ") +
               cudaGetErrorString(counted) + ".";
    }
    for (size_t i = 0; i < all.size(); ++i) {
        const int d = all[i];
        if (d < 0 || d >= ndev) {
            return "cuDSS MG mode: the model groups name CUDA devices " + listed +
                   ", but this process sees " + std::to_string(ndev) +
                   " CUDA device(s), ordinals 0 to " + std::to_string(ndev - 1) + ".";
        }
        for (size_t j = 0; j < i; ++j) {
            if (all[j] == d) {
                return "cuDSS MG mode: the model groups " + listed + " name CUDA device " +
                       std::to_string(d) + " twice; each device belongs to one group.";
            }
        }
    }
    size_t found = all.size();
    for (size_t i = 0; i < all.size(); ++i) {
        if (all[i] == device) found = i;
    }
    if (found == all.size()) {
        return "cuDSS MG mode: the solve runs on CUDA device " + std::to_string(device) +
               ", which is in none of its model groups " + listed +
               "; cuDSS factors on the devices of the group that holds the device the solve "
               "runs on. Place the solve's arrays on a device of a model group, for example "
               "with jax.device_put.";
    }
    const size_t start = found - found % size;
    for (size_t i = 0; i < size; ++i) {
        group->push_back(all[start + (found - start + i) % size]);
    }
    return std::string();
}

// The path of cuDSS's threading layer library (libcudss_mtlayer_gomp.so.N) in the
// cuDSS wheel, registered by the Python loader at import (set_threading_layer);
// empty when the wheel ships none.
static std::mutex& ThreadingLayerMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::string& ThreadingLayerPath() {
    static std::string path;
    return path;
}

static std::string ThreadingLayer() {
    std::lock_guard<std::mutex> lock(ThreadingLayerMutex());
    return ThreadingLayerPath();
}

// Diagnostic knobs for attributing a factorization's cost, read from the
// environment at a handle's cold start. Each one leaves cuDSS's own default
// in place when unset, so nothing changes unless a variable is exported:
//   DIFFSOUP_CUDSS_HOST_NTHREADS=<k>  load cuDSS's threading layer (the library
//       named by CUDSS_THREADING_LIB, else the one of the cuDSS wheel that the
//       loader registered) and run the host side of the analysis with k
//       threads; the reordering is single-threaded otherwise;
//   DIFFSOUP_CUDSS_ND_NLEVELS=<k>     minimum nested-dissection depth;
//   DIFFSOUP_CUDSS_ND_UBFACTOR=<k>    nested-dissection balance factor, in
//       percent (NVIDIA recommends 2 to 5 for multi-device handles);
//   DIFFSOUP_CUDSS_HYBRID_EXECUTE=1   cuDSS hybrid host/device execution.
static int EnvInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return std::atoi(v);
}

static ffi::Error ApplyDiagnosticConfig(cudssHandle_t handle, cudssConfig_t config) {
    cudssStatus_t st;
    const int nthreads = EnvInt("DIFFSOUP_CUDSS_HOST_NTHREADS", 0);
    if (nthreads > 0) {
        const char* lib = std::getenv("CUDSS_THREADING_LIB");
        const std::string name =
            (lib != nullptr && lib[0] != '\0') ? std::string(lib) : ThreadingLayer();
        if (name.empty()) {
            return ffi::Error(
                ffi::ErrorCode::kFailedPrecondition,
                "DIFFSOUP_CUDSS_HOST_NTHREADS asks for cuDSS's threading layer, but the cuDSS "
                "wheel holds no libcudss_mtlayer_gomp.so* and CUDSS_THREADING_LIB is unset; set "
                "CUDSS_THREADING_LIB to the path of the threading layer library, or unset "
                "DIFFSOUP_CUDSS_HOST_NTHREADS.");
        }
        CUDSS_CALL_AND_CHECK(cudssSetThreadingLayer(handle, name.c_str()), st, "cudssSetThreadingLayer");
        int v = nthreads;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_HOST_NTHREADS, &v, sizeof(v)), st, "cudssConfigSet host_nthreads");
    }
    const int nlevels = EnvInt("DIFFSOUP_CUDSS_ND_NLEVELS", 0);
    if (nlevels > 0) {
        int v = nlevels;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_ND_NLEVELS, &v, sizeof(v)), st, "cudssConfigSet nd_nlevels");
    }
    const int ubfactor = EnvInt("DIFFSOUP_CUDSS_ND_UBFACTOR", 0);
    if (ubfactor > 0) {
        int v = ubfactor;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_ND_UBFACTOR, &v, sizeof(v)), st, "cudssConfigSet nd_ubfactor");
    }
    if (EnvInt("DIFFSOUP_CUDSS_HYBRID_EXECUTE", 0) > 0) {
        int v = 1;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &v, sizeof(v)), st, "cudssConfigSet hybrid_execute");
    }
    return ffi::Error::Success();
}

// Facts about a built factor, read from cuDSS after a cold factorization:
// nnz(L + U), the factorization's flop count, and the number of superpanels.
// A query the library does not answer leaves its field at -1.
struct FactorFacts {
    int64_t lu_nnz = -1;
    int64_t flops = -1;
    int64_t nsuperpanels = -1;
};

static FactorFacts QueryFactorFacts(cudssHandle_t handle, cudssData_t data) {
    FactorFacts f;
    size_t written = 0;
    int64_t v64 = -1;
    if (cudssDataGet(handle, data, CUDSS_DATA_LU_NNZ, &v64, sizeof(v64), &written) == CUDSS_STATUS_SUCCESS) {
        f.lu_nnz = v64;
    }
    v64 = -1;
    written = 0;
    if (cudssDataGet(handle, data, CUDSS_DATA_FLOPS, &v64, sizeof(v64), &written) == CUDSS_STATUS_SUCCESS) {
        f.flops = v64;
    }
    // The superpanel count has the matrix index type (32-bit here).
    int32_t v32 = -1;
    written = 0;
    if (cudssDataGet(handle, data, CUDSS_DATA_NSUPERPANELS, &v32, sizeof(v32), &written) == CUDSS_STATUS_SUCCESS) {
        f.nsuperpanels = v32;
    }
    return f;
}

static void QueryMemoryEstimates(cudssHandle_t handle, cudssData_t data, int64_t out[6]) {
    int64_t est[16];
    for (int i = 0; i < 16; ++i) est[i] = -1;
    size_t written = 0;
    if (cudssDataGet(handle, data, CUDSS_DATA_MEMORY_ESTIMATES, est, sizeof(est), &written) == CUDSS_STATUS_SUCCESS) {
        for (int i = 0; i < 6; ++i) out[i] = est[i];
    }
}

// The refusal of a hybrid device memory limit of `limit_mb` MB (MiB) below the device memory
// cuDSS needs for an analysed factorization (CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, `minimum_bytes`,
// which includes the solve's transient workspace): empty when the limit is unset or suffices.
// cuDSS rejects a smaller limit at factorization, so the callers check it after the analysis,
// before they set the limit.
static std::string HybridLimitRefusal(int64_t minimum_bytes, int64_t limit_mb) {
    const int64_t mib = 1024 * 1024;
    if (limit_mb <= 0 || minimum_bytes <= limit_mb * mib) return std::string();
    const std::string needed = std::to_string((minimum_bytes + mib - 1) / mib);
    return "cuDSS hybrid memory needs at least " + needed +
           " MB of device memory for this factorization; hybrid_device_memory_limit_mb is " +
           std::to_string(limit_mb) + ". Set it to " + needed + " or more, or leave it unset.";
}

// cuDSS reads CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT per device: under a
// multi-device handle the limit "must be set for each device separately by
// calling cudaSetDevice() prior to cudssConfigSet()" (cuDSS documentation), so
// one call caps only the dispatch device and leaves the others at the library
// default. Set the cap on every device of the set, then restore the caller's.
// VERIFY: cuDSS 1.0 still reads the MG limit per device (its notes require an explicit
// config topology for multi-GPU hybrid memory, which the MG path sets).
static ffi::Error SetHybridDeviceLimit(
    cudssConfig_t config, int64_t limit_bytes,
    const std::vector<int>& mg_indices, int restore_device
) {
    cudssStatus_t st;
    if (mg_indices.empty()) {
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &limit_bytes, sizeof(limit_bytes)), st, "cudssConfigSet hybrid_limit");
        return ffi::Error::Success();
    }
    for (int d : mg_indices) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &limit_bytes, sizeof(limit_bytes)), st, "cudssConfigSet hybrid_limit (per MG device)");
    }
    CUDA_CHECK(cudaSetDevice(restore_device));
    return ffi::Error::Success();
}

// Make `config` hold solve mode `mode` (CUDSS_CONFIG_SOLVE_MODE) when `*current`, the mode it
// holds, differs: transpose for a solve with the transpose of the factored matrix, normal for
// every other phase. cuDSS takes the change on the configuration of factored data (measured with
// cuDSS 1.0.0.35 on one device).
static ffi::Error SetConfigSolveMode(cudssConfig_t config, cudssSolveMode_t mode,
                                     cudssSolveMode_t* current) {
    if (*current == mode) return ffi::Error::Success();
    cudssStatus_t st;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(config, CUDSS_CONFIG_SOLVE_MODE, &mode, sizeof(mode)), st, "cudssConfigSet solve_mode");
    *current = mode;
    return ffi::Error::Success();
}

// Device span of one phase on the caller's stream, measured with CUDA events
// when DIFF_SOUP_CUDSS_CUDA_EVENTS is set. The end event is waited on, so the
// measurement serializes the host against the phase; it exists to attribute
// wall time between host-side and device-side work, not for production runs.
// Single-device handles only: a multi-device phase completes host-synchronously
// on cuDSS's own streams, so its wall time already is its device span.
struct PhaseTimer {
    bool active = false;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t stream = nullptr;
    PhaseTimer(cudaStream_t s, bool enabled) : stream(s) {
        if (!enabled) return;
        if (cudaEventCreate(&start) != cudaSuccess) return;
        if (cudaEventCreate(&stop) != cudaSuccess) { cudaEventDestroy(start); start = nullptr; return; }
        active = cudaEventRecord(start, stream) == cudaSuccess;
    }
    // Milliseconds between the events, or 0 when inactive.
    double finish() {
        double ms = 0.0;
        if (active && cudaEventRecord(stop, stream) == cudaSuccess && cudaEventSynchronize(stop) == cudaSuccess) {
            float f = 0.0f;
            if (cudaEventElapsedTime(&f, start, stop) == cudaSuccess) ms = f;
        }
        active = false;
        return ms;
    }
    ~PhaseTimer() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

// Helper function for data types ==============================================
// cuDSS matrix descriptors and communication callbacks take CUDA's own data type enum,
// cudaDataType_t (library_types.h): the value type of an XLA element type, and CUDA_R_32I for
// the int32 row offsets and column indices.
template <ffi::DataType T> cudaDataType_t get_cuda_data_type();
template<> cudaDataType_t get_cuda_data_type<ffi::F32>() { return CUDA_R_32F; }
template<> cudaDataType_t get_cuda_data_type<ffi::F64>() { return CUDA_R_64F; }
template<> cudaDataType_t get_cuda_data_type<ffi::C64>() { return CUDA_C_32F; }
template<> cudaDataType_t get_cuda_data_type<ffi::C128>() { return CUDA_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

#include "mgmn.h"

// Status codes of the per-call solver's status output (those of diff_soup.linear.SolveStatus).
constexpr int32_t kStatusSuccess = 0;
constexpr int32_t kStatusSingular = 1;

// The status of a factor from its diagonal (CUDSS_DATA_DIAG), read after a numeric phase, by
// the rules of the one-device contract handlers. The factor is SINGULAR when the diagonal holds
// a non-finite or a zero entry: with pivot perturbation off, an exactly zero pivot stays zero and
// the entries computed from it become infinite or nan. A float64 factor whose matrix was
// equilibrated by matching (`matched`) is also SINGULAR when its smallest pivot magnitude is at
// most 16 unit roundoffs of its largest: after equilibration a pivot that small means the matrix
// is singular to working precision (an exactly dependent column leaves a remainder of about one
// unit roundoff rather than a zero pivot). Factors without matching keep the caller's units, and
// at float32 the pivot ratios of regular and singular matrices overlap, so those take only the
// zero and non-finite conditions. SUCCESS otherwise. cuDSS reports zero in CUDSS_DATA_INFO even
// for an exactly singular matrix, so the diagonal is what detects one. The read waits for the
// work queued on the handle's stream.
template <ffi::DataType T>
static ffi::Error DiagonalStatus(cudssHandle_t handle, cudssData_t data, int64_t n,
                                 bool matched, int32_t* status) {
    using scalar_t = typename get_native_data_type<T>::type;
    std::vector<scalar_t> diagonal(static_cast<size_t>(n));
    const size_t bytes = diagonal.size() * sizeof(scalar_t);
    size_t written = 0;
    cudssStatus_t st;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, data, CUDSS_DATA_DIAG, diagonal.data(), bytes, &written), st, "cudssDataGet diag");
    if (written != bytes) {
        return ffi::Error::Internal("cuDSS wrote " + std::to_string(written) +
                                    " bytes for the diagonal of a factor; its " +
                                    std::to_string(n) + " entries take " +
                                    std::to_string(bytes) + " bytes.");
    }
    double smallest = std::numeric_limits<double>::infinity();
    double largest = 0.0;
    for (const scalar_t& entry : diagonal) {
        const double magnitude = static_cast<double>(std::abs(entry));
        if (!std::isfinite(magnitude)) {
            *status = kStatusSingular;
            return ffi::Error::Success();
        }
        smallest = std::min(smallest, magnitude);
        largest = std::max(largest, magnitude);
    }
    constexpr bool f64 = std::is_same_v<scalar_t, double>;
    const double unit_roundoff = std::ldexp(1.0, -53);
    const bool tiny = matched && f64 && smallest <= 16.0 * unit_roundoff * largest;
    *status = (smallest == 0.0 || tiny) ? kStatusSingular : kStatusSuccess;
    return ffi::Error::Success();
}

// Per-device cuDSS state =====================================================
// When shard_map distributes across devices, each device needs its own
// cuDSS handle, topology buffers, and matrix descriptors. The DeviceState
// holds everything for ONE device. The outer CudssState holds a map from
// device_id to DeviceState, populated lazily on first execute per device.
template <ffi::DataType T>
struct DeviceState {
    int device_id = -1;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    int32_t* p_offsets = nullptr;
    int32_t* p_columns = nullptr;
    int64_t n = 0;
    int64_t nnz = 0;
    int64_t nrhs = 0;
    int64_t call_count = 0;
    cudssDeviceMemHandler_t mem_handler{};
    bool mem_handler_initialized = false;
    // MG mode: the CUDA ordinals the factor is spread over, this device
    // first. Empty in single-device mode.
    std::vector<int> mg_indices;
    // The status of the factor from its latest numeric phase (DiagonalStatus).
    int32_t status = kStatusSuccess;
    // The CUDSS_CONFIG_SOLVE_MODE `config` holds: the transpose mode is set for a solve with the
    // transpose of the factored matrix, and normal again before a numeric phase.
    cudssSolveMode_t solve_mode = CUDSS_SOLVE_MODE_NORMAL;

    ~DeviceState() {
        if (handle) {
            cudaSetDevice(device_id);
            if (A) cudssMatrixDestroy(A);
            if (b) cudssMatrixDestroy(b);
            if (x) cudssMatrixDestroy(x);
            if (data) cudssDataDestroy(handle, data);
            if (config) cudssConfigDestroy(config);
            cudssDestroy(handle);
        }
        if (p_offsets) { cudaSetDevice(device_id); cudaFree(p_offsets); }
        if (p_columns) { cudaSetDevice(device_id); cudaFree(p_columns); }
    }
};

// Outer state: shared across devices, holds config + per-device map
template <ffi::DataType T>
struct CudssState {
    std::shared_ptr<mgmn::Factor> mgmn_factor;
    static xla::ffi::TypeId id;

    // Config (set at instantiation, shared across devices)
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    int64_t reordering_alg = -1;
    int64_t use_matching = -1;
    int64_t matching_alg = -1;
    int64_t use_device_mem_handler = 0;
    int64_t use_superpanels = -1;
    // Hybrid memory mode: hybrid_mode != 0 enables host-spilled
    // factorization (CUDSS_CONFIG_HYBRID_MEMORY_MODE, set BEFORE the ANALYSIS
    // phase). hybrid_device_memory_limit_mb > 0 caps device factor residency
    // (CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, bytes), set AFTER the analysis
    // once CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN (the device floor) is known; a
    // limit below the floor is refused. The floor is stashed back on the
    // device state so the host layer can read it.
    int64_t hybrid_mode = 0;
    int64_t hybrid_device_memory_limit_mb = -1;
    // deterministic != 0 sets CUDSS_CONFIG_DETERMINISTIC_MODE
    // before ANALYSIS (bit-wise run-to-run reproducibility on same-arch GPUs;
    // single-GPU, single-RHS, hybrid-off, IR-off per docs — the Python layer
    // fences the combinations). Default 0 = config never set (the cuDSS
    // default behavior, byte-identical to never touching the option).
    int64_t deterministic = 0;
    // Multi-GPU (MG) mode: mg_group_size >= 2 routes the cold start through
    // cudssCreateMg, with CUDSS_CONFIG_DEVICE_COUNT/_INDICES set on the config
    // before the analysis; 0 = one device (plain cudssCreate). mg_devices holds
    // the CUDA ordinals of every model group of the layout's mesh in this
    // process, group after group, each in the mesh's order. Inputs stay whole
    // on the dispatch device (cuDSS MG takes no distributed input) and cuDSS
    // spreads the factor across the devices of the group that holds the
    // dispatch device. cuDSS requires the calling device to be the first
    // device index, so that group is rotated to start at it; a dispatch device
    // in no group is refused.
    std::vector<int> mg_devices;
    int64_t mg_group_size = 0;
    cudaDataType_t cuda_dtype = get_cuda_data_type<T>();

    // Whether the factorization runs a matching, and so equilibrates the
    // matrix: an explicit matching_alg other than NONE, else use_matching > 0;
    // cuDSS runs none for a positive definite matrix type.
    bool Matched() const {
        if (mtype == CUDSS_MTYPE_SPD || mtype == CUDSS_MTYPE_HPD) return false;
        if (matching_alg >= 0) return matching_alg != CUDSS_MATCHING_ALG_NONE;
        return use_matching > 0;
    }

    // Per-device state map: device_id -> DeviceState
    std::mutex mu;
    std::unordered_map<int, std::unique_ptr<DeviceState<T>>> devices;

    DeviceState<T>* get_or_create(int dev_id) {
        std::lock_guard<std::mutex> lock(mu);
        auto it = devices.find(dev_id);
        if (it != devices.end()) return it->second.get();
        auto ds = std::make_unique<DeviceState<T>>();
        ds->device_id = dev_id;
        auto* ptr = ds.get();
        devices[dev_id] = std::move(ds);
        return ptr;
    }
};

template <> ffi::TypeId CudssState<ffi::F32>::id = {};
template <> ffi::TypeId CudssState<ffi::F64>::id = {};
template <> ffi::TypeId CudssState<ffi::C64>::id = {};
template <> ffi::TypeId CudssState<ffi::C128>::id = {};

// instantiation ===============================================================
template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<CudssState<T>>> CudssInstantiate(
    ffi::Span<const int32_t> csr_offsets,
    ffi::Span<const int32_t> csr_columns,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t ir_nsteps,
    const int64_t instance_id,
    const int64_t refit_mode,
    const int64_t reordering_alg,
    const int64_t use_matching,
    const int64_t matching_alg,
    const int64_t use_device_mem_handler,
    const int64_t use_superpanels,
    const int64_t hybrid_mode,
    const int64_t hybrid_device_memory_limit_mb,
    const int64_t deterministic,
    ffi::Span<const int32_t> mg_devices,
    const int64_t mg_group_size,
    const int64_t mgmn_group,
    const int64_t mgmn_input,
    const int64_t solve_mode
) {
    std::shared_ptr<const HostTopology> topology_keep;
    if (auto topology_error = ResolveTopology(instance_id, csr_offsets, csr_columns, topology_keep);
        topology_error.failure()) {
        return topology_error;
    }
    // The topology/config attributes are instantiate-time constants. The host
    // copies of the CSR pattern flow through as attributes on every execute
    // call as well (attributes are compile-time HLO constants, not buffers),
    // so we don't need to stash them on the state — Execute will receive the
    // same Spans. They are still accepted here so the Instantiate and Execute
    // bindings share the same attribute set, which keeps the FFI contract
    // aligned.
    (void)csr_offsets;
    (void)csr_columns;
    (void)ir_nsteps;
    (void)refit_mode;
    (void)solve_mode;
    (void)mgmn_input;
    (void)device_id;  // ignored — device detected from stream at execute time
    ResetStats(instance_id);
    auto state = std::make_unique<CudssState<T>>();

    if (mtype_id == 0) state->mtype = CUDSS_MTYPE_GENERAL;
    else if (mtype_id == 1) state->mtype = CUDSS_MTYPE_SYMMETRIC;
    else if (mtype_id == 2) state->mtype = CUDSS_MTYPE_HERMITIAN;
    else if (mtype_id == 3) state->mtype = CUDSS_MTYPE_SPD;
    else if (mtype_id == 4) state->mtype = CUDSS_MTYPE_HPD;
    else return ffi::ErrorOr<std::unique_ptr<CudssState<T>>>(ffi::Error::InvalidArgument("Invalid mtype_id."));

    if (mview_id == 0) state->mview = CUDSS_MVIEW_FULL;
    else if (mview_id == 1) state->mview = CUDSS_MVIEW_UPPER;
    else if (mview_id == 2) state->mview = CUDSS_MVIEW_LOWER;
    else return ffi::ErrorOr<std::unique_ptr<CudssState<T>>>(ffi::Error::InvalidArgument("Invalid mview_id."));

    state->reordering_alg = reordering_alg;
    state->use_matching = use_matching;
    state->matching_alg = matching_alg;
    state->use_device_mem_handler = (use_device_mem_handler != 0) ? 1 : 0;
    state->use_superpanels = use_superpanels;
    state->hybrid_mode = (hybrid_mode != 0) ? 1 : 0;
    state->hybrid_device_memory_limit_mb = hybrid_device_memory_limit_mb;
    state->deterministic = (deterministic != 0) ? 1 : 0;
    state->mg_devices.assign(mg_devices.begin(), mg_devices.end());
    state->mg_group_size = mg_group_size;

    return ffi::ErrorOr<std::unique_ptr<CudssState<T>>>(std::move(state));
}

// execution ===================================================================
template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,
    CudssState<T>* state,
    ffi::Buffer<T> b_values_buf,
    ffi::Buffer<T> csr_values_buf,
    ffi::ResultBuffer<T> out_values_buf,
    ffi::ResultBuffer<ffi::S32> status_buf,
    ffi::Span<const int32_t> csr_offsets,
    ffi::Span<const int32_t> csr_columns,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t ir_nsteps,
    const int64_t instance_id,
    const int64_t refit_mode,
    const int64_t reordering_alg,
    const int64_t use_matching,
    const int64_t matching_alg,
    const int64_t use_device_mem_handler,
    const int64_t use_superpanels,
    const int64_t hybrid_mode,
    const int64_t hybrid_device_memory_limit_mb,
    const int64_t deterministic,
    ffi::Span<const int32_t> mg_devices,
    const int64_t mg_group_size,
    const int64_t mgmn_group,
    const int64_t mgmn_input,
    const int64_t solve_mode
) {
    std::shared_ptr<const HostTopology> topology_keep;
    if (auto topology_error = ResolveTopology(instance_id, csr_offsets, csr_columns, topology_keep);
        topology_error.failure()) {
        return topology_error;
    }
    if (solve_mode != 0 && solve_mode != 1) {
        return ffi::Error::InvalidArgument(
            "cuDSS solve_mode is 0 (solve with the factored matrix) or 1 (with its transpose); "
            "got " + std::to_string(solve_mode) + ".");
    }
    // The solve runs with the transpose of the factored matrix in cuDSS's transpose solve mode;
    // every numeric phase runs in normal mode.
    // VERIFY: the transpose solve mode under MG and MGMN (measured on one device only).
    const cudssSolveMode_t mode =
        solve_mode == 1 ? CUDSS_SOLVE_MODE_TRANSPOSE : CUDSS_SOLVE_MODE_NORMAL;
    if (mgmn_group) {
        // The input of the group's factorization: 0 puts the matrix and the right-hand side on
        // its first process (central), 1 gives each process its own block of rows (distributed).
        if (mgmn_input != 0 && mgmn_input != 1) {
            return ffi::Error::InvalidArgument(
                "cuDSS mgmn_input is 0 (central input) or 1 (distributed input); got " +
                std::to_string(mgmn_input) + ".");
        }
        try {
        auto& distributed = mgmn::GetRaw<T>(instance_id, mgmn_group, state->mgmn_factor);
        std::lock_guard<std::mutex> lock(distributed.group->mutex);
        // The matching as on one device: an explicit matching_alg, else NONE when matching is
        // turned off, AUTO when it is turned on, and cuDSS's default when neither is given.
        const int matching = matching_alg >= 0 ? static_cast<int>(matching_alg)
                             : use_matching == 0 ? static_cast<int>(CUDSS_MATCHING_ALG_NONE)
                             : use_matching > 0  ? static_cast<int>(CUDSS_MATCHING_ALG_AUTO)
                                                 : -1;
        mgmn::Refactor<T>(distributed, stream, csr_values_buf.typed_data(), csr_offsets, csr_columns,
            mtype_id, mview_id, ir_nsteps, reordering_alg, matching,
            hybrid_mode, hybrid_device_memory_limit_mb, use_superpanels, mgmn_input == 1);

            mgmn::Solve<T>(distributed, stream, b_values_buf.typed_data(), out_values_buf->typed_data(), b_values_buf.element_count(), mode);
            // The factor's diagonal is not read across processes: the status output reports
            // SUCCESS and the caller's status comes from the values.
            const int32_t success = kStatusSuccess;
            mgmn::Check(cudaMemcpyAsync(status_buf->typed_data(), &success, sizeof(success),
                                        cudaMemcpyHostToDevice, stream), "write MGMN status");
            mgmn::Check(cudaStreamSynchronize(stream), "MGMN status synchronization");
            return ffi::Error::Success();
        } catch (const std::invalid_argument& error) {
            return ffi::Error::InvalidArgument(error.what());
        } catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
    }

    (void)mtype_id;
    (void)mview_id;
    (void)ir_nsteps;
    (void)reordering_alg;
    (void)use_matching;
    (void)matching_alg;
    (void)use_device_mem_handler;
    (void)use_superpanels;
    // hybrid_mode / hybrid_device_memory_limit_mb / deterministic are read off
    // ``state`` (set at Instantiate); the per-call attributes are accepted only
    // to keep the Instantiate/Execute attribute sets aligned.
    (void)hybrid_mode;
    (void)hybrid_device_memory_limit_mb;
    (void)deterministic;
    (void)mg_devices;
    (void)mg_group_size;

    // Detect actual device from XLA's context (not the static attribute)
    int actual_device = 0;
    cudaGetDevice(&actual_device);

    const bool do_refit = (refit_mode != 0);
    using scalar_t = typename get_native_data_type<T>::type;
    SolverDebugStats delta_stats;
    delta_stats.execute_calls = 1;
    delta_stats.solve_calls = 1;
    if (do_refit) delta_stats.refit_calls = 1;
    const auto total_t0 = Clock::now();

    // Get per-device state (created on first call per device)
    DeviceState<T>* ds = state->get_or_create(actual_device);

    const int64_t rhs_elements = b_values_buf.element_count();
    const int64_t out_elements = out_values_buf->element_count();
    if (rhs_elements != out_elements) {
        return ffi::Error::InvalidArgument("RHS and output buffers must have matching element counts");
    }

    // Topology is carried in attribute spans now, not in runtime buffers.
    // Both come from the HLO custom-call attribute set and point into
    // compile-time HOST memory.
    const int64_t offsets_size = static_cast<int64_t>(csr_offsets.size());
    const int64_t columns_size = static_cast<int64_t>(csr_columns.size());

    const bool trace = ShouldTrace(ds->call_count);
    if (trace) {
        printf("[cudss_debug] dev=%d state=%p ds=%p call=%lld do_refit=%d\n",
               actual_device, static_cast<void*>(state), static_cast<void*>(ds),
               static_cast<long long>(ds->call_count), do_refit ? 1 : 0);
        fflush(stdout);
    }

    const bool mg_on = state->mg_group_size >= 2;
    if (ds->call_count == 0) {
        delta_stats.cold_calls = 1;
        // === COLD START (per device) ===
        ds->n = offsets_size - 1;
        ds->nnz = columns_size;
        if (ds->n <= 0 || ds->nnz <= 0) {
            return ffi::Error::InvalidArgument("Invalid CSR topology dimensions");
        }
        if (rhs_elements <= 0 || rhs_elements % ds->n != 0) {
            return ffi::Error::InvalidArgument("RHS buffer has invalid size for matrix dimension");
        }
        ds->nrhs = rhs_elements / ds->n;

        // MG mode: a single-process multi-GPU handle over the model group of
        // mg_devices that holds the dispatch device, rotated to start at it.
        // The documented incompatibilities and a device set this process
        // cannot use are refused before anything is allocated.
        std::vector<int> mg_indices;
        if (mg_on) {
            if (state->deterministic != 0) {
                return ffi::Error::InvalidArgument(
                    "cuDSS DETERMINISTIC_MODE is single-GPU only; "
                    "it cannot be combined with MG mode.");
            }
            if (state->use_device_mem_handler != 0) {
                return ffi::Error::InvalidArgument(
                    "The custom device memory handler is untested under cuDSS "
                    "MG mode; disable use_device_mem_handler for MG solves.");
            }
            const std::string refusal = ModelGroupOf(
                state->mg_devices, state->mg_group_size, actual_device, &mg_indices);
            if (!refusal.empty()) return ffi::Error::InvalidArgument(refusal);
        }

        // Allocate persistent topology ON THIS DEVICE and copy from the
        // host-side attribute spans (Host → Device). XLA FFI's Span exposes
        // ``begin()`` (not ``.data()``) for the underlying pointer.
        CUDA_CHECK(cudaMalloc(&ds->p_offsets, offsets_size * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ds->p_columns, columns_size * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpyAsync(ds->p_offsets, csr_offsets.begin(),
                   offsets_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ds->p_columns, csr_columns.begin(),
                   columns_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        cudssStatus_t st;
        if (mg_on) {
            {
                ffi::Error perr = EnableMgPeerAccess(mg_indices, actual_device);
                if (perr.failure()) return perr;
            }
            CUDSS_CALL_AND_CHECK(
                cudssCreateMg(&ds->handle,
                              static_cast<int>(mg_indices.size()),
                              mg_indices.data()),
                st, "cudssCreateMg");
            ds->mg_indices = mg_indices;
        } else {
            CUDSS_CALL_AND_CHECK(cudssCreate(&ds->handle), st, "cudssCreate");
        }
        CUDSS_CALL_AND_CHECK(cudssSetStream(ds->handle, stream), st, "cudssSetStream");

        if (state->use_device_mem_handler != 0) {
            if (!ds->mem_handler_initialized) {
                ds->mem_handler.ctx = nullptr;
                ds->mem_handler.device_alloc = WrappedDeviceAlloc;
                ds->mem_handler.device_free = WrappedDeviceFree;
                std::strncpy(ds->mem_handler.name, "diff_soup async cuda allocator", CUDSS_ALLOCATOR_NAME_LEN);
                ds->mem_handler.name[CUDSS_ALLOCATOR_NAME_LEN - 1] = '\0';
                ds->mem_handler_initialized = true;
            }
            CUDSS_CALL_AND_CHECK(cudssSetDeviceMemHandler(ds->handle, &ds->mem_handler), st, "cudssSetDeviceMemHandler");
        }

        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&ds->config), st, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(ds->handle, &ds->data), st, "cudssDataCreate");
        {
            ffi::Error derr = ApplyDiagnosticConfig(ds->handle, ds->config);
            if (derr.failure()) return derr;
        }

        // Algorithms are plain int config values. Matching is controlled
        // solely by CUDSS_CONFIG_MATCHING_ALG (CUDSS_MATCHING_ALG_NONE disables it).
        if (state->reordering_alg >= 0) {
            int alg = static_cast<int>(state->reordering_alg);
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)), st, "cudssConfigSet reordering");
        }
        // Fold use_matching intent into MATCHING_ALG. If the host requested
        // a specific matching_alg, honor it; else if matching was explicitly
        // disabled (use_matching == 0), pass MATCHING_ALG_NONE. use_matching > 0
        // with no explicit alg -> AUTO.
        if (state->matching_alg >= 0) {
            int alg = static_cast<int>(state->matching_alg);
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)), st, "cudssConfigSet matching_alg");
        } else if (state->use_matching == 0) {
            int alg = CUDSS_MATCHING_ALG_NONE;
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)), st, "cudssConfigSet matching_alg(none)");
        } else if (state->use_matching > 0) {
            int alg = CUDSS_MATCHING_ALG_AUTO;
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_MATCHING_ALG, &alg, sizeof(alg)), st, "cudssConfigSet matching_alg(auto)");
        }
        if (state->use_superpanels >= 0) {
            int v = static_cast<int>(state->use_superpanels);
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_USE_SUPERPANELS, &v, sizeof(v)), st, "cudssConfigSet superpanels");
        }

        // The descriptors point at this call's XLA buffers, in MG mode as on
        // one device; warm calls re-point them at the buffers of their call.
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&ds->b, ds->n, ds->nrhs, ds->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), st, "cudssMatrixCreateDn b");
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&ds->x, ds->n, ds->nrhs, ds->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), st, "cudssMatrixCreateDn x");
        // cudssMatrixCreateCsr takes the offset, index and value types. Our row
        // offsets and column indices are both int32 -> (offsetType, indexType,
        // valueType) = (CUDA_R_32I, CUDA_R_32I, cuda_dtype).
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&ds->A, ds->n, ds->n, ds->nnz,
            ds->p_offsets, NULL, ds->p_columns, csr_values_buf.typed_data(),
            CUDA_R_32I, CUDA_R_32I, state->cuda_dtype, state->mtype, state->mview, state->base), st, "cudssMatrixCreateCsr");

        int ir = static_cast<int>(ir_nsteps);
        CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)), st, "cudssConfigSet ir");

        // Hybrid memory mode: enable host-spilled factorization BEFORE the
        // ANALYSIS phase. The optional device-factor cap is set after it, once
        // the floor it must not undercut is known (below).
        if (state->hybrid_mode != 0) {
            int hm = 1;
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hm, sizeof(hm)), st, "cudssConfigSet hybrid_mode");
        }

        // Opt-in bit-wise deterministic kernels. Set BEFORE ANALYSIS: cuDSS
        // refuses a change after it. Python fences the unsupported
        // combinations (hybrid / multi-RHS / IR / COLAMD).
        if (state->deterministic != 0) {
            int dm = 1;
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_DETERMINISTIC_MODE, &dm, sizeof(dm)), st, "cudssConfigSet deterministic");
        }

        // MG mode: the documented activation pairing — the Mg handle
        // (above) plus DEVICE_COUNT/DEVICE_INDICES on the config BEFORE
        // ANALYSIS (advanced_features "MG Mode" + simple_mg_mode sample).
        if (mg_on) {
            int devc = static_cast<int>(mg_indices.size());
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_DEVICE_COUNT, &devc, sizeof(devc)), st, "cudssConfigSet device_count");
            CUDSS_CALL_AND_CHECK(cudssConfigSet(ds->config, CUDSS_CONFIG_DEVICE_INDICES, mg_indices.data(), devc * static_cast<int>(sizeof(int))), st, "cudssConfigSet device_indices");
        }

        if constexpr (ZeroFillsSolveOutput<T>()) {
            ffi::Error zerr = ZeroSolveOutput(
                out_values_buf->typed_data(), static_cast<size_t>(out_elements) * sizeof(scalar_t),
                stream, mg_on);
            if (zerr.failure()) return zerr;
        }
        if (mg_on) {
            // MG work runs on cuDSS-managed internal per-device streams that
            // are not ordered against the XLA stream producing this call's
            // input buffers; block until their producers finish before any
            // phase reads them. Local (single-device) mode is untouched: its
            // work rides the XLA stream and needs no synchronization.
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        {
            const auto t0 = Clock::now();
            CUDSS_CALL_AND_CHECK(cudssExecute(ds->handle, CUDSS_PHASE_ANALYSIS,
                ds->config, ds->data, ds->A, ds->x, ds->b), st, "cudssExecute analysis");
            delta_stats.analysis_ms += ElapsedMs(t0);
        }

        // Query the hybrid device-memory floor after analysis (valid only in
        // hybrid mode). Surfaced to the host through the stats channel.
        // VERIFY: under MG the floor read on the dispatch device bounds every device's.
        if (state->hybrid_mode != 0) {
            int64_t hybrid_min = -1;
            size_t written = 0;
            cudssStatus_t qs = cudssDataGet(ds->handle, ds->data,
                CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &hybrid_min,
                sizeof(hybrid_min), &written);
            delta_stats.hybrid_device_memory_min =
                (qs == CUDSS_STATUS_SUCCESS) ? hybrid_min : -1;
        }
        // Post-analysis memory estimates (16xint64; slots
        // 0-5 meaningful). Read-only diagnostics — enables a later auto-hybrid
        // policy instead of the hardcoded device-cap dial. Failure is benign
        // (slots stay -1).
        {
            int64_t est[16];
            for (int i = 0; i < 16; ++i) est[i] = -1;
            size_t written = 0;
            cudssStatus_t qs = cudssDataGet(ds->handle, ds->data,
                CUDSS_DATA_MEMORY_ESTIMATES, est, sizeof(est), &written);
            if (qs == CUDSS_STATUS_SUCCESS) {
                for (int i = 0; i < 6; ++i) delta_stats.memory_estimates[i] = est[i];
            }
            // Bank the analysis outcome now: a factorization that runs out of
            // memory returns before the call's stats are accumulated, and the
            // estimate is what sizes the fix.
            SolverDebugStats after_analysis;
            after_analysis.analysis_ms = delta_stats.analysis_ms;
            after_analysis.hybrid_device_memory_min = delta_stats.hybrid_device_memory_min;
            for (int i = 0; i < 6; ++i) after_analysis.memory_estimates[i] = delta_stats.memory_estimates[i];
            AccumulateStats(instance_id, after_analysis);
            delta_stats.analysis_ms = 0.0;
        }
        // The hybrid device-factor cap, set after the analysis: cuDSS's floor
        // includes the solve's workspace once the analysis has run, and a cap
        // below it is refused here with both numbers (cuDSS would reject it at
        // factorization). An unreadable floor leaves the check to cuDSS.
        if (state->hybrid_mode != 0 && state->hybrid_device_memory_limit_mb > 0) {
            if (delta_stats.hybrid_device_memory_min >= 0) {
                const std::string refusal = HybridLimitRefusal(
                    delta_stats.hybrid_device_memory_min, state->hybrid_device_memory_limit_mb);
                if (!refusal.empty()) return ffi::Error::InvalidArgument(refusal);
            }
            int64_t lim = state->hybrid_device_memory_limit_mb * 1024 * 1024;
            ffi::Error lerr = SetHybridDeviceLimit(ds->config, lim, mg_indices, actual_device);
            if (lerr.failure()) return lerr;
        }
        {
            const auto t0 = Clock::now();
            st = cudssExecute(ds->handle, CUDSS_PHASE_FACTORIZATION,
                ds->config, ds->data, ds->A, ds->x, ds->b);
            if (st != CUDSS_STATUS_SUCCESS) {
                printf("[diff_soup cudss] FAILED: cuDSS call returned status = %d, details: cudssExecute factorization\n", st);
                return CudssCallFailed(st, "cudssExecute factorization", delta_stats.memory_estimates[1]);
            }
            delta_stats.factorization_ms += ElapsedMs(t0);
            const FactorFacts facts = QueryFactorFacts(ds->handle, ds->data);
            delta_stats.lu_nnz = facts.lu_nnz;
            delta_stats.factor_flops = facts.flops;
            delta_stats.nsuperpanels = facts.nsuperpanels;
            ffi::Error derr = DiagonalStatus<T>(ds->handle, ds->data, ds->n, state->Matched(), &ds->status);
            if (derr.failure()) return derr;
        }
        {
            ffi::Error merr = SetConfigSolveMode(ds->config, mode, &ds->solve_mode);
            if (merr.failure()) return merr;
            const auto t0 = Clock::now();
            CUDSS_CALL_AND_CHECK(cudssExecute(ds->handle, CUDSS_PHASE_SOLVE,
                ds->config, ds->data, ds->A, ds->x, ds->b), st, "cudssExecute solve");
            delta_stats.solve_phase_ms += ElapsedMs(t0);
        }
    }
    else {
        delta_stats.warm_calls = 1;
        // === WARM START (per device) ===
        if (offsets_size != ds->n + 1 || columns_size != ds->nnz) {
            return ffi::Error::InvalidArgument("CSR topology changed");
        }
        if (rhs_elements <= 0 || rhs_elements % ds->n != 0) {
            return ffi::Error::InvalidArgument("RHS buffer has invalid size for matrix dimension");
        }
        int64_t current_nrhs = rhs_elements / ds->n;

        cudssStatus_t st;
        CUDSS_CALL_AND_CHECK(cudssSetStream(ds->handle, stream), st, "cudssSetStream");

        // Re-point the descriptors at this call's buffers; a change of the
        // number of right-hand sides rebuilds the RHS and solution descriptors.
        CUDSS_CALL_AND_CHECK(cudssMatrixSetCsrPointers(ds->A, ds->p_offsets, NULL, ds->p_columns,
            csr_values_buf.typed_data()), st, "update_pointers A");

        if (current_nrhs != ds->nrhs) {
            if (ds->b) { CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(ds->b), st, "destroy b"); ds->b = nullptr; }
            if (ds->x) { CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(ds->x), st, "destroy x"); ds->x = nullptr; }
            ds->nrhs = current_nrhs;
            CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&ds->b, ds->n, ds->nrhs, ds->n,
                b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), st, "recreate b");
            CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&ds->x, ds->n, ds->nrhs, ds->n,
                out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), st, "recreate x");
        } else {
            CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(ds->b, b_values_buf.typed_data()), st, "update b");
            CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(ds->x, out_values_buf->typed_data()), st, "update x");
        }

        if constexpr (ZeroFillsSolveOutput<T>()) {
            ffi::Error zerr = ZeroSolveOutput(
                out_values_buf->typed_data(),
                static_cast<size_t>(out_elements) * sizeof(scalar_t),
                stream, mg_on);
            if (zerr.failure()) return zerr;
        }
        if (mg_on) {
            // See the cold-start note: the XLA-stream producers of the
            // values/RHS buffers finish before the MG phases run.
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (do_refit) {
            ffi::Error merr =
                SetConfigSolveMode(ds->config, CUDSS_SOLVE_MODE_NORMAL, &ds->solve_mode);
            if (merr.failure()) return merr;
            const auto t0 = Clock::now();
            CUDSS_CALL_AND_CHECK(cudssExecute(ds->handle, CUDSS_PHASE_REFACTORIZATION,
                ds->config, ds->data, ds->A, ds->x, ds->b), st, "cudssExecute refactorization");
            delta_stats.refactorization_ms += ElapsedMs(t0);
            ffi::Error derr = DiagonalStatus<T>(ds->handle, ds->data, ds->n, state->Matched(), &ds->status);
            if (derr.failure()) return derr;
        }
        {
            ffi::Error merr = SetConfigSolveMode(ds->config, mode, &ds->solve_mode);
            if (merr.failure()) return merr;
            const auto t0 = Clock::now();
            CUDSS_CALL_AND_CHECK(cudssExecute(ds->handle, CUDSS_PHASE_SOLVE,
                ds->config, ds->data, ds->A, ds->x, ds->b), st, "cudssExecute solve");
            delta_stats.solve_phase_ms += ElapsedMs(t0);
        }
    }

    // The status of the factor this call solved with, on the caller's stream.
    // A pageable host source is staged before cudaMemcpyAsync returns.
    {
        const int32_t status = ds->status;
        CUDA_CHECK(cudaMemcpyAsync(status_buf->typed_data(), &status, sizeof(status),
                                   cudaMemcpyHostToDevice, stream));
    }

    delta_stats.total_ms += ElapsedMs(total_t0);
    AccumulateStats(instance_id, delta_stats);
    ds->call_count++;

    if (trace) {
        PrintMemInfo("after_execute");
        fflush(stdout);
    }

    return ffi::Error::Success();
}

#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<ffi::Span<const int32_t>>("csr_offsets") \
            .Attr<ffi::Span<const int32_t>>("csr_columns") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("ir_nsteps") \
            .Attr<int64_t>("instance_id") \
            .Attr<int64_t>("refit_mode") \
            .Attr<int64_t>("reordering_alg") \
            .Attr<int64_t>("use_matching") \
            .Attr<int64_t>("matching_alg") \
            .Attr<int64_t>("use_device_mem_handler") \
            .Attr<int64_t>("use_superpanels") \
            .Attr<int64_t>("hybrid_mode") \
            .Attr<int64_t>("hybrid_device_memory_limit_mb") \
            .Attr<int64_t>("deterministic") \
            .Attr<ffi::Span<const int32_t>>("mg_devices") \
            .Attr<int64_t>("mg_group_size") \
            .Attr<int64_t>("mgmn_group") \
            .Attr<int64_t>("mgmn_input") \
            .Attr<int64_t>("solve_mode")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecute##TypeName, CudssExecute<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<ffi::Span<const int32_t>>("csr_offsets") \
            .Attr<ffi::Span<const int32_t>>("csr_columns") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("ir_nsteps") \
            .Attr<int64_t>("instance_id") \
            .Attr<int64_t>("refit_mode") \
            .Attr<int64_t>("reordering_alg") \
            .Attr<int64_t>("use_matching") \
            .Attr<int64_t>("matching_alg") \
            .Attr<int64_t>("use_device_mem_handler") \
            .Attr<int64_t>("use_superpanels") \
            .Attr<int64_t>("hybrid_mode") \
            .Attr<int64_t>("hybrid_device_memory_limit_mb") \
            .Attr<int64_t>("deterministic") \
            .Attr<ffi::Span<const int32_t>>("mg_devices") \
            .Attr<int64_t>("mg_group_size") \
            .Attr<int64_t>("mgmn_group") \
            .Attr<int64_t>("mgmn_input") \
            .Attr<int64_t>("solve_mode"));

DEFINE_CUDSS_FFI_HANDLERS(f32, ffi::F32);
DEFINE_CUDSS_FFI_HANDLERS(f64, ffi::F64);
DEFINE_CUDSS_FFI_HANDLERS(c64, ffi::C64);
DEFINE_CUDSS_FFI_HANDLERS(c128, ffi::C128);

// XLA owns the handler state it instantiates and needs the type info (its
// deleter) alongside the type id to destroy that state with the executable.
#define EXPORT_CUDSS_HANDLERS(m, TypeName, DataType) \
    m.def("type_id_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(&CudssState<DataType>::id)); \
    }); \
    m.def("type_info_" #TypeName, []() { \
        static const XLA_FFI_TypeInfo info = ffi::MakeTypeInfo<CudssState<DataType>>(); \
        return nb::capsule(const_cast<XLA_FFI_TypeInfo*>(&info)); \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    });

#include "contract_handlers.inc"

NB_MODULE(_native_solver, m) {
    mgmn::Export(m);
    m.def(
        "cudss_version",
        []() {
            // cudssGetProperty(libraryPropertyType, int*) answers MAJOR_VERSION, MINOR_VERSION
            // and PATCH_LEVEL, as the other CUDA math libraries do.
            int major = -1;
            int minor = -1;
            int patch = -1;
            if (cudssGetProperty(MAJOR_VERSION, &major) != CUDSS_STATUS_SUCCESS ||
                cudssGetProperty(MINOR_VERSION, &minor) != CUDSS_STATUS_SUCCESS ||
                cudssGetProperty(PATCH_LEVEL, &patch) != CUDSS_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "cudssGetProperty failed: the cuDSS library loaded in this process does not "
                    "report its version.");
            }
            return nb::make_tuple(major, minor, patch);
        },
        "The (major, minor, patch) version of the cuDSS library loaded in this process.");
    m.def(
        "set_threading_layer",
        [](const std::string& path) {
            std::lock_guard<std::mutex> lock(ThreadingLayerMutex());
            ThreadingLayerPath() = path;
        },
        "Record the path of cuDSS's threading layer library (libcudss_mtlayer_gomp.so.N), which "
        "DIFFSOUP_CUDSS_HOST_NTHREADS loads when CUDSS_THREADING_LIB is unset.");
    m.def(
        "cudss_built_version",
        []() {
            return nb::make_tuple(static_cast<int>(DIFF_SOUP_CUDSS_MAJOR),
                                  static_cast<int>(DIFF_SOUP_CUDSS_MINOR),
                                  static_cast<int>(DIFF_SOUP_CUDSS_PATCH));
        },
        "The (major, minor, patch) version of the cuDSS headers the extension was built against.");
    m.def(
        "register_topology",
        [](int64_t instance_id,
           nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> offsets,
           nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> columns) {
            RegisterTopology(instance_id, offsets.data(), offsets.shape(0), columns.data(),
                             columns.shape(0));
        },
        "Store the CSR pattern (int32 offsets and columns) of a solver instance on the host, "
        "so its FFI calls can carry empty pattern attributes.");
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);

    m.def("debug_get_stats", [](int64_t instance_id) {
        SolverDebugStats s = GetStatsSnapshot(instance_id);
        nb::dict d;
        d["execute_calls"] = s.execute_calls;
        d["handler_calls"] = s.handler_calls;
        d["analysis_calls"] = s.analysis_calls;
        d["factorization_calls"] = s.factorization_calls;
        d["refactorization_calls"] = s.refactorization_calls;
        d["solve_phase_calls"] = s.solve_phase_calls;
        // Kept as an explicit zero-valued compatibility key so qualification
        // can assert that the physically removed serialized path is absent.
        d["serialized_fallback_calls"] = 0;
        d["cold_calls"] = s.cold_calls;
        d["warm_calls"] = s.warm_calls;
        d["refit_calls"] = s.refit_calls;
        d["solve_calls"] = s.solve_calls;
        // Host/enqueue steady_clock spans (not device execution time).
        d["total_host_ms"] = s.total_ms;
        d["set_stream_host_ms"] = s.set_stream_ms;
        d["set_csr_host_ms"] = s.set_csr_ms;
        d["set_values_host_ms"] = s.set_values_ms;
        d["analysis_host_ms"] = s.analysis_ms;
        d["factorization_host_ms"] = s.factorization_ms;
        d["refactorization_host_ms"] = s.refactorization_ms;
        d["solve_phase_host_ms"] = s.solve_phase_ms;
        // CUDA-event device spans.
        d["analysis_cuda_ms"] = s.analysis_cuda_ms;
        d["factorization_cuda_ms"] = s.factorization_cuda_ms;
        d["refactorization_cuda_ms"] = s.refactorization_cuda_ms;
        d["solve_phase_cuda_ms"] = s.solve_phase_cuda_ms;
        d["hybrid_device_memory_min"] = s.hybrid_device_memory_min;
        // Post-analysis memory estimates, slots 0-5
        // (perm dev / peak dev / perm host / peak host / hybrid min dev /
        // hybrid max host, bytes; -1 = not queried).
        nb::list est;
        for (int i = 0; i < 6; ++i) est.append(s.memory_estimates[i]);
        d["memory_estimates"] = est;
        // Facts about the last cold factor: nnz(L + U), factorization flops,
        // superpanel count (-1 = not queried).
        d["lu_nnz"] = s.lu_nnz;
        d["factor_flops"] = s.factor_flops;
        d["nsuperpanels"] = s.nsuperpanels;
        return d;
    });

    m.def("debug_reset_stats", [](int64_t instance_id) {
        ResetStats(instance_id);
    });
    BindContract(m);
}
