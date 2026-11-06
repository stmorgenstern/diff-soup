// SuperLU_MT sparse LU on the host, exposed to JAX through XLA FFI handlers.
//
// Objects:
//   * an analysis per sparsity pattern: the compressed-column structure of
//     the operator, the map from the caller's coordinate entries to it, the
//     Dirichlet rows, and the fill-reducing column ordering from get_perm_c;
//   * a slot per factor: the numeric LU factors (L, U), the row and column
//     permutations, the elimination tree and supernode partition
//     (SuperLU_MT's refactorization structure), and a copy of the values the
//     factors were built from;
//   * a site per lowered factor or refactor call: the slots that call writes
//     are keyed at the site by the context id the call carries at run time.
//     A factor call picks its slot by (context, slot token), so the lanes
//     of a value batch (which carry distinct tokens) get one slot each, every
//     context gets its own, and a repeated execution reuses them. A refactor
//     call picks its slot by (context, the handle it refactors), and
//     refactors in place when that handle is already one of its own slots,
//     so a loop that carries a factor keeps one slot.
//
// Slots belong to a context (diff_soup.linear.SolveContext): a factor handle
// is the pair (context id, slot id), every slot records its context, and
// context_release(id) frees every slot of that context and records the id as
// released. A call that meets a released context, a slot that does not exist
// or a slot of another context fails with an error. Solve and refactor calls
// carry the id of their analysis as the attribute "analysis_id" and fail on
// a slot built for another analysis. Sites belong to no context and live for
// the process.
//
// Every handler takes a fresh XLA token as its first operand and returns one as
// its first result, and takes the factor's order word as its last operand without
// reading it. The lowering passes the call's status as the next order word, so a
// refactorization of a slot never runs before a solve whose returned factor it refactors.
//
// A slot is refactored in place with pdgstrf(refact = YES), which keeps the
// column ordering, the elimination tree and the storage of L and U, when the
// previous factorization in the process used the same analysis. SuperLU_MT
// keeps the storage sizes of its last factorization in process-wide state
// and trusts them on a refactorization, so a slot whose pattern differs from
// that last factorization is rebuilt from the analysis's column ordering
// with refact = NO instead.
//
// Solves use dgstrs with trans = NOTRANS or TRANS on the same factors, with
// every right-hand side in one column-major block. A solve whose values
// operand differs from the values its slot was last factored with rebuilds
// the slot from the operand first, so a solve never reads a factor built
// from other values.
//
// Complex values: every handler has a complex128 twin (the targets ending in
// _c128) that factors with pzgstrf and solves with zgstrs on the same
// analysis, column ordering and slot machinery. A transposed solve is the
// plain transpose A^T (trans = TRANS), never the conjugate transpose. A slot
// records whether it holds real or complex factors, and a call of the other
// kind factors it afresh, as a call with other values does. SuperLU_MT keeps
// the storage sizes of each precision's last factorization in that
// precision's own state, so the refactorization rule above is applied per
// kind.
//
// Shared storage: a factor that empty_factor(..., storage=...) started names a slot keyed by
// (context id, device ordinal, analysis, storage key, empty-lane marker) instead of a site's
// slot, shared by every call site and program on that device; its calls carry the storage key.
// The devices of one program (host devices placed over a mesh) each hold their own slot, so
// neither trusts a factorization the other made. Its refactor runs in that slot
// whichever site runs it and, with a false predicate, factors the values it keeps when the slot
// holds other ones. Its solve uses the slot as it stands when a refactor of the slot ran earlier
// in the same execution of the program (XLA's run id), and compares values as above otherwise.
// A site is named by a key the lowering computes from the call and created on the first call
// that carries it.
//
// SuperLU_MT preallocates L and U and ends the process when a factorization
// outgrows them, so the storage it allocates is the worst case for any row
// pivoting (see ComputeFillBounds), supplied through sp_ienv; a failed
// allocation surfaces as an error of the FFI call.
//
// pdgstrf runs on every core the process may use (its CPU affinity, capped
// by OMP_NUM_THREADS when that is set). One mutex serialises every call into
// the library; the Python-facing functions release the GIL before taking it.

#include <sched.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/ffi/api/ffi.h"

#include "slu_mt_ddefs.h"
#include "superlu_z_prototypes.h"
#include "slot_registry.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {

// Result codes shared with diff_soup.linear.SolveStatus.
constexpr int32_t kSuccess = 0;
constexpr int32_t kSingular = 1;
constexpr int32_t kNonfinite = 2;

int ComputeNprocs() {
    int count = 0;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        count = CPU_COUNT(&set);
    }
    if (count <= 0) {
        count = omp_get_max_threads();
    }
    const char* env = std::getenv("OMP_NUM_THREADS");
    if (env != nullptr && *env != '\0') {
        const int requested = std::atoi(env);
        if (requested > 0 && requested < count) {
            count = requested;
        }
    }
    return std::max(1, count);
}

int Nprocs() {
    static const int nprocs = ComputeNprocs();
    return nprocs;
}

struct Stats {
    int64_t factorizations = 0;    // pdgstrf or pzgstrf with refact = NO
    int64_t refactorizations = 0;  // pdgstrf or pzgstrf with refact = YES
    int64_t rebuilds = 0;          // factorizations started by a solve
    int64_t solves = 0;            // dgstrs or zgstrs calls
    int64_t nonfinite = 0;         // factor requests refused for non-finite values
    int64_t released_context_errors = 0;  // calls refused because their context was released
};

// The pattern (slot_registry.h), the column ordering and the factorization's parameters.
struct Analysis : slot_registry::HostPattern<int_t> {
    std::vector<int_t> perm_c;         // column ordering from get_perm_c
    int_t panel_size = 0;
    int_t relax = 0;
    double pivot_threshold = 1.0;
    // Worst-case lengths of U's values/subscripts and of L's subscripts for
    // any row pivoting (0 until the first factorization computes them).
    int64_t ucol_bound = 0;
    int64_t lsub_bound = 0;
};

struct Slot;
void FreeStorage(Slot& slot);

struct Slot : slot_registry::SlotBase<Analysis> {
    bool has_storage = false;          // L, U and the refactorization structure exist
    bool has_values = false;           // `values` holds the operand of the last factor request
    bool complex = false;              // values, csc, L and U are complex128
    int32_t status = kSingular;
    // The values and the compressed-column values, two doubles per entry when complex.
    std::vector<double> values;
    std::vector<double> csc;
    std::vector<int_t> perm_c;
    std::vector<int_t> perm_r;
    superlumt_options_t options{};
    SuperMatrix L{};
    SuperMatrix U{};

    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() { FreeStorage(*this); }
};

using Site = slot_registry::SiteBase<Analysis>;

struct Registry
    : slot_registry::RegistryBase<Analysis, Slot, Site, Stats, slot_registry::HostStorageKey> {
    static constexpr const char* kName = "SuperLU";
    static constexpr const char* kStorage = "host storage";
    int64_t last_factored_analysis = 0;          // of the latest pdgstrf
    int64_t last_factored_complex_analysis = 0;  // of the latest pzgstrf
};

Registry& Reg() { return slot_registry::Instance<Registry>(); }

using slot_registry::CheckAnalysis;
using slot_registry::CheckContext;
using slot_registry::CheckValues;
using slot_registry::Doubles;
using slot_registry::FindHeld;
using slot_registry::kComplex;
using slot_registry::SameValues;
using slot_registry::SiteFor;
using slot_registry::SiteSlot;
using slot_registry::StorageSlot;

void FreeStorage(Slot& slot) {
    if (!slot.has_storage) {
        return;
    }
    Destroy_SuperNode_SCP(&slot.L);
    Destroy_CompCol_NCP(&slot.U);
    SUPERLU_FREE(slot.options.etree);
    SUPERLU_FREE(slot.options.colcnt_h);
    SUPERLU_FREE(slot.options.part_super_h);
    slot.options.etree = nullptr;
    slot.options.colcnt_h = nullptr;
    slot.options.part_super_h = nullptr;
    slot.L = SuperMatrix{};
    slot.U = SuperMatrix{};
    slot.has_storage = false;
}

// Storage sizes handed to SuperLU_MT through sp_ienv(7) and sp_ienv(8) while
// one factorization allocates (0 selects the library's own guesses). Written
// and read under the library mutex.
int_t g_fill_ucol = 0;
int_t g_fill_lsub = 0;

// Factor storage could not be allocated (pdgstrf info > n).
constexpr int32_t kAllocationFailed = -1;

// Worst-case storage for any row pivoting, from the column counts of the
// Cholesky factor Lc of A^T A in the factorization's column order (George and
// Ng): struct(L) and struct(U^T) lie in struct(Lc). L's subscripts need at
// most nnz(Lc) + n entries; U is stored in dense segments per supernode,
// which adds at most relax - 1 explicit zeros per segment in a relaxed
// supernode, so U needs at most relax * nnz(Lc) + n. SuperLU_MT's default
// guesses (50 and 30 times nnz(A)) are exceeded by high-fill orderings, and
// running out of preallocated storage ends the process, so the bounds replace
// them. Runs after pdgstrf_init(refact = NO), which leaves the post-ordered
// column permutation and elimination tree in `slot`.
void ComputeFillBounds(Analysis& a, const Slot& slot) {
    const int_t n = a.n;
    std::vector<int_t> iperm(static_cast<size_t>(n));
    std::vector<int_t> invp(static_cast<size_t>(n));
    std::vector<int_t> colcnt(static_cast<size_t>(n));
    std::vector<int_t> part_ata(static_cast<size_t>(n));
    std::vector<int_t> part_h(static_cast<size_t>(n));
    for (int_t i = 0; i < n; ++i) {
        iperm[i] = i;
        invp[slot.perm_c[i]] = i;
    }
    int_t nlnz = 0;
    qrnzcnt(n, static_cast<int_t>(a.rowind.size()), a.colptr.data(), a.rowind.data(),
            iperm.data(), invp.data(), const_cast<int_t*>(slot.perm_c.data()),
            slot.options.etree, colcnt.data(), &nlnz, part_ata.data(), part_h.data());
    const int64_t limit = std::numeric_limits<int_t>::max();
    const int64_t lc = nlnz > 0 ? static_cast<int64_t>(nlnz) : limit;
    const int64_t width = std::max<int64_t>(1, a.relax);
    a.lsub_bound = std::min(limit, lc + n);
    a.ucol_bound = std::min(limit, width * lc + n);
}

// Factor `values` (the caller's coordinate entries; two doubles per entry, real then imaginary
// part, when `Complex`) into `slot`. Caller holds the library mutex. Returns the slot's status,
// or kAllocationFailed.
template <bool Complex>
int32_t FactorSlot(Slot& slot, const double* values) {
    constexpr int64_t width = Complex ? 2 : 1;  // doubles per entry
    Registry& reg = Reg();
    Analysis& a = *slot.analysis;
    const int64_t nnz = a.nnz_entries;
    slot.values.assign(values, values + width * nnz);
    slot.has_values = true;

    for (int64_t k = 0; k < width * nnz; ++k) {
        if (!std::isfinite(values[k])) {
            // A slot whose storage is of the other kind cannot be refactored as this kind.
            if (slot.complex != Complex) FreeStorage(slot);
            slot.complex = Complex;
            slot.status = kNonfinite;
            ++reg.stats.nonfinite;
            return slot.status;
        }
    }

    const int64_t nnz_csc = static_cast<int64_t>(a.rowind.size());
    slot.csc.assign(static_cast<size_t>(width * nnz_csc), 0.0);
    for (int64_t k = 0; k < nnz; ++k) {
        if (!a.zeroed[k]) {
            if constexpr (Complex) {
                slot.csc[2 * a.entry_to_csc[k]] += values[2 * k];
                slot.csc[2 * a.entry_to_csc[k] + 1] += values[2 * k + 1];
            } else {
                slot.csc[a.entry_to_csc[k]] += values[k];
            }
        }
    }
    for (int64_t p : a.unit_diagonal) {
        slot.csc[width * p] += 1.0;
    }

    // SuperLU_MT keeps the storage sizes of each precision's last factorization apart, so a
    // refactorization needs a slot of this kind whose analysis that precision factored last.
    const int64_t last_factored =
        Complex ? reg.last_factored_complex_analysis : reg.last_factored_analysis;
    const bool refact = slot.has_storage && slot.complex == Complex && last_factored == a.id;
    if (!refact) {
        FreeStorage(slot);
        slot.complex = Complex;
        slot.perm_c = a.perm_c;  // pdgstrf_init post-orders it in place
        slot.perm_r.assign(static_cast<size_t>(a.n), 0);
        slot.options = superlumt_options_t{};
    }

    const int_t nprocs = static_cast<int_t>(Nprocs());
    SuperMatrix A{};
    SuperMatrix AC{};
    if constexpr (Complex) {
        zCreate_CompCol_Matrix(&A, a.n, a.n, static_cast<int_t>(nnz_csc),
                               reinterpret_cast<doublecomplex*>(slot.csc.data()), a.rowind.data(),
                               a.colptr.data(), SLU_NC, SLU_Z, SLU_GE);
    } else {
        dCreate_CompCol_Matrix(&A, a.n, a.n, static_cast<int_t>(nnz_csc), slot.csc.data(),
                               a.rowind.data(), a.colptr.data(), SLU_NC, SLU_D, SLU_GE);
    }
    Gstat_t gstat{};
    StatAlloc(a.n, nprocs, a.panel_size, a.relax, &gstat);
    StatInit(a.n, nprocs, &gstat);

    if constexpr (Complex) {
        pzgstrf_init(nprocs, DOFACT, NOTRANS, refact ? YES : NO, a.panel_size, a.relax,
                     a.pivot_threshold, NO, 0.0, slot.perm_c.data(), slot.perm_r.data(), nullptr,
                     0, &A, &AC, &slot.options, &gstat);
    } else {
        pdgstrf_init(nprocs, DOFACT, NOTRANS, refact ? YES : NO, a.panel_size, a.relax,
                     a.pivot_threshold, NO, 0.0, slot.perm_c.data(), slot.perm_r.data(), nullptr,
                     0, &A, &AC, &slot.options, &gstat);
    }

    if (!refact) {
        if (a.ucol_bound == 0) {
            ComputeFillBounds(a, slot);
        }
        g_fill_ucol = static_cast<int_t>(a.ucol_bound);
        g_fill_lsub = static_cast<int_t>(a.lsub_bound);
    }

    // The OpenMP build runs one busy-waiting worker per loop iteration, so the
    // team must have exactly nprocs threads.
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(nprocs));
    int_t info = 0;
    if constexpr (Complex) {
        pzgstrf(&slot.options, &AC, slot.perm_r.data(), &slot.L, &slot.U, &gstat, &info);
    } else {
        pdgstrf(&slot.options, &AC, slot.perm_r.data(), &slot.L, &slot.U, &gstat, &info);
    }
    g_fill_ucol = 0;
    g_fill_lsub = 0;

    Destroy_CompCol_Permuted(&AC);
    Destroy_SuperMatrix_Store(&A);
    StatFree(&gstat);

    if (info > a.n) {
        // Storage allocation failed before factoring: L and U were not built.
        SUPERLU_FREE(slot.options.etree);
        SUPERLU_FREE(slot.options.colcnt_h);
        SUPERLU_FREE(slot.options.part_super_h);
        slot.options = superlumt_options_t{};
        slot.L = SuperMatrix{};
        slot.U = SuperMatrix{};
        slot.has_storage = false;
        slot.has_values = false;
        slot.status = kSingular;
        return kAllocationFailed;
    }

    slot.has_storage = true;
    if constexpr (Complex) {
        reg.last_factored_complex_analysis = a.id;
    } else {
        reg.last_factored_analysis = a.id;
    }
    if (refact) {
        ++reg.stats.refactorizations;
    } else {
        ++reg.stats.factorizations;
    }
    slot.status = (info == 0) ? kSuccess : kSingular;
    return slot.status;
}

ffi::Error AllocationError(const Analysis& a) {
    const double gib = static_cast<double>(a.ucol_bound) * (sizeof(double) + sizeof(int_t)) /
                       (1024.0 * 1024.0 * 1024.0);
    return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                      "SuperLU could not allocate the LU factor storage of a " +
                          std::to_string(a.n) + " x " + std::to_string(a.n) +
                          " operator (worst-case U storage about " + std::to_string(gib) +
                          " GiB). Release unused SolveContexts, choose a fill-reducing ordering "
                          "(options={'ordering': 'colamd'}), or run on a host with more memory.");
}

// Factor into `slot` and write its status, or return the allocation error.
template <bool Complex>
ffi::Error FactorInto(Slot& slot, const double* values, int32_t* status) {
    const int32_t result = FactorSlot<Complex>(slot, values);
    if (result == kAllocationFailed) {
        return AllocationError(*slot.analysis);
    }
    *status = result;
    return ffi::Error::Success();
}

int64_t SlotBytes(const Slot& slot) {
    int64_t bytes = static_cast<int64_t>(slot.values.capacity() + slot.csc.capacity()) *
                    static_cast<int64_t>(sizeof(double));
    bytes += static_cast<int64_t>(slot.perm_c.capacity() + slot.perm_r.capacity()) *
             static_cast<int64_t>(sizeof(int_t));
    if (slot.has_storage) {
        const auto* lstore = static_cast<const SCPformat*>(slot.L.Store);
        const auto* ustore = static_cast<const NCPformat*>(slot.U.Store);
        const int64_t nnz = static_cast<int64_t>(lstore->nnz) + static_cast<int64_t>(ustore->nnz);
        const int64_t value_size = slot.complex ? 2 * sizeof(double) : sizeof(double);
        bytes += nnz * (value_size + static_cast<int64_t>(sizeof(int_t)));
        bytes += 3 * static_cast<int64_t>(slot.analysis->n) * static_cast<int64_t>(sizeof(int_t));
    }
    return bytes;
}

// --------------------------------------------------------------------------
// XLA FFI handlers
// --------------------------------------------------------------------------

// Every handler takes the factor's order word as its last buffer operand and never reads it:
// the word makes the call a data dependency of the operation that produced it, and the call's
// status result becomes the next order word.
template <ffi::DataType T>
ffi::Error FactorImpl(ffi::RunId run, int32_t device, ffi::Token, ffi::Buffer<T> values,
                      ffi::Buffer<ffi::S32> key,
                      ffi::Buffer<ffi::S32> context_id, ffi::Buffer<ffi::S32> /*order*/,
                      ffi::Result<ffi::Token>,
                      ffi::ResultBuffer<ffi::S32> handle_context,
                      ffi::ResultBuffer<ffi::S32> handle_slot, ffi::ResultBuffer<ffi::S32> status,
                      int64_t site_id, int64_t analysis_id, int64_t storage) {
    const int32_t context = context_id.typed_data()[0];
    const int64_t token = key.typed_data()[0];
    Registry& reg = Reg();
    std::lock_guard<std::mutex> lock(reg.mutex);
    if (auto err = CheckContext(reg, context); err.failure()) {
        return err;
    }
    Site* site = nullptr;
    if (auto err = SiteFor(reg, site_id, analysis_id, &site); err.failure()) {
        return err;
    }
    // A factor of shared storage (the fill of an empty factor that names it) goes to the
    // storage's slot, keyed by the empty handle's slot id; any other to the site's slot.
    Slot* slot = nullptr;
    if (auto err = storage != 0 ? StorageSlot(reg, *site, context, device, storage, token, &slot)
                                : SiteSlot(reg, *site, context, token, &slot);
        err.failure()) {
        return err;
    }
    if (auto err = CheckValues(reg, *slot, values); err.failure()) {
        return err;
    }
    if (auto err = FactorInto<kComplex<T>>(*slot, Doubles(values), status->typed_data());
        err.failure()) {
        return err;
    }
    handle_context->typed_data()[0] = context;
    if (storage != 0) slot->trusted_run = run.run_id;
    handle_slot->typed_data()[0] = slot->id;
    return ffi::Error::Success();
}

template <ffi::DataType T>
ffi::Error RefactorImpl(ffi::RunId run, ffi::Token, ffi::Buffer<T> values,
                        ffi::Buffer<ffi::PRED> predicate, ffi::Buffer<ffi::S32> held_context,
                        ffi::Buffer<ffi::S32> held_slot, ffi::Buffer<ffi::S32> /*order*/,
                        ffi::Result<ffi::Token>,
                        ffi::ResultBuffer<ffi::S32> handle_context,
                        ffi::ResultBuffer<ffi::S32> handle_slot,
                        ffi::ResultBuffer<ffi::S32> status,
                        int64_t site_id,
                        int64_t analysis_id, int64_t storage) {
    const bool refactor = predicate.typed_data()[0];
    const int32_t context = held_context.typed_data()[0];
    const int32_t held_id = held_slot.typed_data()[0];
    Registry& reg = Reg();
    std::lock_guard<std::mutex> lock(reg.mutex);
    Slot* held = nullptr;
    if (auto err = FindHeld(reg, context, held_id, &held); err.failure()) {
        return err;
    }
    if (auto err = CheckAnalysis(reg, *held, analysis_id); err.failure()) {
        return err;
    }
    handle_context->typed_data()[0] = context;
    // A factor of shared storage makes the storage hold the values it keeps (the values operand
    // carries them): with a false predicate it is refactored only when the storage holds other
    // values, for instance at the entry of a replayed segment.
    if (!refactor && (storage == 0 || SameValues(*held, Doubles(values), kComplex<T>))) {
        if (storage != 0) held->trusted_run = run.run_id;
        handle_slot->typed_data()[0] = held_id;
        status->typed_data()[0] = held->status;
        return ffi::Error::Success();
    }
    Site* site = nullptr;
    if (auto err = SiteFor(reg, site_id, analysis_id, &site); err.failure()) {
        return err;
    }
    // In place when the held factor is already this site's or names shared storage; otherwise
    // the site's slot for that held factor, so other holders of it keep it.
    Slot* slot = held;
    if (held->site != site_id && storage == 0) {
        if (auto err = SiteSlot(reg, *site, context, held_id, &slot); err.failure()) {
            return err;
        }
    }
    if (auto err = CheckValues(reg, *slot, values); err.failure()) {
        return err;
    }
    if (auto err = FactorInto<kComplex<T>>(*slot, Doubles(values), status->typed_data());
        err.failure()) {
        return err;
    }
    if (storage != 0) slot->trusted_run = run.run_id;
    handle_slot->typed_data()[0] = slot->id;
    return ffi::Error::Success();
}

template <ffi::DataType T>
ffi::Error ApplyImpl(ffi::RunId run, ffi::Token, ffi::Buffer<T> values,
                     ffi::Buffer<ffi::S32> held_context,
                     ffi::Buffer<ffi::S32> held_slot, ffi::Buffer<T> b,
                     ffi::Buffer<ffi::S32> /*order*/,
                     ffi::Result<ffi::Token>, ffi::ResultBuffer<T> x,
                     ffi::ResultBuffer<ffi::S32> status,
                     bool transpose, int64_t analysis_id, int64_t storage) {
    const int32_t context = held_context.typed_data()[0];
    const int32_t id = held_slot.typed_data()[0];
    Registry& reg = Reg();
    std::lock_guard<std::mutex> lock(reg.mutex);
    Slot* slot = nullptr;
    if (auto err = FindHeld(reg, context, id, &slot); err.failure()) {
        return err;
    }
    if (auto err = CheckAnalysis(reg, *slot, analysis_id); err.failure()) {
        return err;
    }
    if (auto err = CheckValues(reg, *slot, values); err.failure()) {
        return err;
    }
    const int_t n = slot->analysis->n;
    const int64_t total = static_cast<int64_t>(b.element_count());
    if (n == 0 || total % n != 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "SuperLU right-hand side has " + std::to_string(total) +
                              " entries, which is not a whole number of columns of length " +
                              std::to_string(n) + ".");
    }
    const int_t nrhs = static_cast<int_t>(total / n);

    // A solve of a factor that names shared storage (storage != 0) uses the storage as it
    // stands when a refactor of it ran earlier in this execution (that refactor made it hold
    // the matrix of the factor it returned); any other solve refactors a slot that holds other
    // values than the ones it carries. Such a refactor (a derivative solve of an older factor
    // runs after a later refactor of the storage) leaves the slot holding this solve's matrix,
    // so the slot stops being trusted: a later solve of this execution compares values again.
    // A trusted shared-storage slot holds the matrix of the factor a refactor of this execution
    // returned, of the kind of that refactor's values, which are this solve's.
    constexpr bool complex = kComplex<T>;
    const bool trusted = storage != 0 && slot->trusted_run == run.run_id;
    if (!trusted && !SameValues(*slot, Doubles(values), complex)) {
        slot->trusted_run = -1;
        ++reg.stats.rebuilds;
        if (FactorSlot<complex>(*slot, Doubles(values)) == kAllocationFailed) {
            return AllocationError(*slot->analysis);
        }
    }

    double* out = reinterpret_cast<double*>(x->typed_data());
    constexpr int64_t width = complex ? 2 : 1;  // doubles per entry
    status->typed_data()[0] = slot->status;
    if (slot->status != kSuccess) {
        std::fill(out, out + width * total, std::numeric_limits<double>::quiet_NaN());
        return ffi::Error::Success();
    }
    std::memcpy(out, Doubles(b), static_cast<size_t>(width * total) * sizeof(double));

    SuperMatrix B{};
    if constexpr (complex) {
        zCreate_Dense_Matrix(&B, n, nrhs, reinterpret_cast<doublecomplex*>(out), n, SLU_DN, SLU_Z,
                             SLU_GE);
    } else {
        dCreate_Dense_Matrix(&B, n, nrhs, out, n, SLU_DN, SLU_D, SLU_GE);
    }
    Gstat_t gstat{};
    const int_t nprocs = static_cast<int_t>(Nprocs());
    StatAlloc(n, nprocs, slot->analysis->panel_size, slot->analysis->relax, &gstat);
    StatInit(n, nprocs, &gstat);
    int_t info = 0;
    // The plain transpose: zgstrs with TRANS solves A^T x = b, never the conjugate transpose.
    if constexpr (complex) {
        zgstrs(transpose ? TRANS : NOTRANS, &slot->L, &slot->U, slot->perm_r.data(),
               slot->perm_c.data(), &B, &gstat, &info);
    } else {
        dgstrs(transpose ? TRANS : NOTRANS, &slot->L, &slot->U, slot->perm_r.data(),
               slot->perm_c.data(), &B, &gstat, &info);
    }
    StatFree(&gstat);
    Destroy_SuperMatrix_Store(&B);
    ++reg.stats.solves;
    if (info != 0) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("SuperLU ") + (complex ? "zgstrs" : "dgstrs") +
                              " rejected argument " + std::to_string(-info) + ".");
    }
    return ffi::Error::Success();
}

}  // namespace

// Replaces the library's sp_ienv (linked statically, so this definition wins):
// the same tuning values as the OpenMP build, with the storage sizes of U
// (ispec 7) and of L's subscripts (ispec 8) taken from the worst-case bounds
// of the factorization being set up.
extern "C" int_t sp_ienv(int_t ispec) {
    switch (ispec) {
        case 1: return 20;   // panel size
        case 2: return 6;    // relaxed supernode size
        case 3: return 200;  // maximum supernode size
        case 4: return 200;  // row block size for 2-D BLAS
        case 5: return 100;  // column block size for 2-D BLAS
        case 6: return -50;  // L values when SuperLU_DYNAMIC_SNODE_STORE is set
        case 7: return g_fill_ucol > 0 ? g_fill_ucol : -50;
        case 8: return g_fill_lsub > 0 ? g_fill_lsub : -30;
        default: return 0;
    }
}

// One factor, refactor and apply handler per value type: NAME is F64 or C128 and TYPE the
// matching ffi::DataType.
#define DIFF_SOUP_SUPERLU_HANDLERS(NAME, TYPE)                                                    \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(kSuperluFactor##NAME, FactorImpl<TYPE>,                         \
                                  ffi::Ffi::Bind()                                                \
                                      .Ctx<ffi::RunId>()            /* the XLA execution */       \
                                      /* the device that runs the call, which keys shared storage */ \
                                      .Ctx<ffi::DeviceOrdinal>()                                  \
                                      .Arg<ffi::Token>()            /* XLA token in */            \
                                      .Arg<ffi::Buffer<TYPE>>()     /* values */                  \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* slot key (call-site token) */ \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* context id */              \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* order word in (unread) */  \
                                      .Ret<ffi::Token>()            /* XLA token out */           \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* handle: context id */      \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* handle: slot */            \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* status */                  \
                                      .Attr<int64_t>("site")                                      \
                                      .Attr<int64_t>("analysis_id")                               \
                                      .Attr<int64_t>("storage"));                                 \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(kSuperluRefactor##NAME, RefactorImpl<TYPE>,                     \
                                  ffi::Ffi::Bind()                                                \
                                      .Ctx<ffi::RunId>()            /* the XLA execution */       \
                                      .Arg<ffi::Token>()            /* XLA token in */            \
                                      .Arg<ffi::Buffer<TYPE>>()     /* values */                  \
                                      .Arg<ffi::Buffer<ffi::PRED>>() /* predicate */              \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* held handle: context id */ \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* held handle: slot */       \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* order word in (unread) */  \
                                      .Ret<ffi::Token>()            /* XLA token out */           \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* handle: context id */      \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* handle: slot */            \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* status */                  \
                                      .Attr<int64_t>("site")                                      \
                                      .Attr<int64_t>("analysis_id")                               \
                                      .Attr<int64_t>("storage"));                                 \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(kSuperluApply##NAME, ApplyImpl<TYPE>,                           \
                                  ffi::Ffi::Bind()                                                \
                                      .Ctx<ffi::RunId>()            /* the XLA execution */       \
                                      .Arg<ffi::Token>()            /* XLA token in */            \
                                      .Arg<ffi::Buffer<TYPE>>()     /* values */                  \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* handle: context id */      \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* handle: slot */            \
                                      .Arg<ffi::Buffer<TYPE>>()     /* right-hand sides */        \
                                      .Arg<ffi::Buffer<ffi::S32>>() /* order word in (unread) */  \
                                      .Ret<ffi::Token>()            /* XLA token out */           \
                                      .Ret<ffi::Buffer<TYPE>>()     /* solution */                \
                                      .Ret<ffi::Buffer<ffi::S32>>() /* status */                  \
                                      .Attr<bool>("transpose")                                    \
                                      .Attr<int64_t>("analysis_id")                               \
                                      .Attr<int64_t>("storage"));

DIFF_SOUP_SUPERLU_HANDLERS(F64, ffi::F64)
DIFF_SOUP_SUPERLU_HANDLERS(C128, ffi::C128)

// --------------------------------------------------------------------------
// Python module
// --------------------------------------------------------------------------

namespace {

using IndexArray = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using MaskArray = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

int64_t AnalysisCreate(int64_t n, IndexArray rows, IndexArray cols, MaskArray zeroed,
                       IndexArray unit_rows, int64_t ordering, int64_t panel_size, int64_t relax,
                       double pivot_threshold) {
    const int64_t nnz = static_cast<int64_t>(rows.shape(0));
    const int32_t* r = rows.data();
    const int32_t* c = cols.data();
    slot_registry::CheckCoordinates<Registry>(n, nnz, r, c, static_cast<int64_t>(cols.shape(0)),
                                              static_cast<int64_t>(zeroed.shape(0)));
    if (ordering < 0 || ordering > 3) {
        throw std::invalid_argument("SuperLU column ordering code must be 0, 1, 2 or 3; got " +
                                    std::to_string(ordering) + ".");
    }

    auto a = std::make_shared<Analysis>();
    a->panel_size = panel_size > 0 ? static_cast<int_t>(panel_size) : sp_ienv(1);
    a->relax = relax > 0 ? static_cast<int_t>(relax) : sp_ienv(2);
    a->pivot_threshold = pivot_threshold;

    {
        nb::gil_scoped_release release;
        slot_registry::BuildPattern<Registry>(*a, n, nnz, r, c, zeroed.data(), unit_rows.data(),
                                              unit_rows.shape(0));
        std::lock_guard<std::mutex> lock(Reg().mutex);
        // get_perm_c reads the structure only; the values pointer is never read.
        std::vector<double> dummy(a->rowind.size(), 0.0);
        SuperMatrix A{};
        dCreate_CompCol_Matrix(&A, a->n, a->n, static_cast<int_t>(a->rowind.size()), dummy.data(),
                               a->rowind.data(), a->colptr.data(), SLU_NC, SLU_D, SLU_GE);
        a->perm_c.assign(static_cast<size_t>(n), 0);
        get_perm_c(static_cast<int_t>(ordering), &A, a->perm_c.data());
        Destroy_SuperMatrix_Store(&A);

        Registry& reg = Reg();
        a->id = reg.next_analysis++;
        reg.analyses[a->id] = a;
    }
    return a->id;
}

}  // namespace

NB_MODULE(_native_superlu, m) {
    m.doc() = "SuperLU_MT sparse LU factorization exposed through XLA FFI handlers.";

    m.def("analysis_create", &AnalysisCreate, nb::arg("n"), nb::arg("rows"), nb::arg("cols"),
          nb::arg("zeroed"), nb::arg("unit_rows"), nb::arg("ordering"), nb::arg("panel_size"),
          nb::arg("relax"), nb::arg("pivot_threshold"),
          "Analyse a sparsity pattern: compressed-column structure, Dirichlet rows and the "
          "column ordering. Returns the analysis id.");
    m.def("analysis_release", &slot_registry::AnalysisRelease<Registry>, nb::arg("analysis_id"),
          "Drop the registry's reference to an analysis; slots that use it keep it alive.");
    m.def("site_create", &slot_registry::SiteCreate<Registry>, nb::arg("analysis_id"),
          "Create a factor site on an analysis with an id of its own (a lowered call names its "
          "site by a key computed from the call, and the site is created on the first call "
          "that carries it); its slots are keyed by the context id each execution carries. "
          "Returns the site id.");
    m.def("context_release", &slot_registry::ContextRelease<Registry>, nb::arg("context_id"),
          "Free every slot of the given SolveContext id and record the id as released, so a "
          "later call that carries it fails.");
    m.def("resident_bytes", &slot_registry::ResidentBytes<Registry>, nb::arg("context_id"),
          "Bytes held by the slots of the given SolveContext id: values, permutations and the "
          "nonzeros of L and U.");
    m.def("nprocs", []() { return Nprocs(); },
          "Threads each factorization uses: the process's CPU affinity count, capped by "
          "OMP_NUM_THREADS when that is set.");
    m.def(
        "stats",
        []() {
            Stats s;
            int64_t live_slots = 0;
            int64_t live_analyses = 0;
            int64_t live_sites = 0;
            {
                nb::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(Reg().mutex);
                s = Reg().stats;
                live_slots = static_cast<int64_t>(Reg().slots.size());
                live_analyses = static_cast<int64_t>(Reg().analyses.size());
                live_sites = static_cast<int64_t>(Reg().sites.size());
            }
            nb::dict d;
            d["factorizations"] = s.factorizations;
            d["refactorizations"] = s.refactorizations;
            d["rebuilds"] = s.rebuilds;
            d["solves"] = s.solves;
            d["nonfinite"] = s.nonfinite;
            d["released_context_errors"] = s.released_context_errors;
            d["live_slots"] = live_slots;
            d["live_analyses"] = live_analyses;
            d["live_sites"] = live_sites;
            return d;
        },
        "Process-wide counters of factorizations (refact = NO), refactorizations "
        "(refact = YES), rebuilds started by a solve, solves, refused non-finite requests, "
        "calls refused for a released context, and live objects.");
    m.def(
        "reset_stats",
        []() {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(Reg().mutex);
            Reg().stats = Stats{};
        },
        "Zero the counters reported by stats().");
    m.def(
        "registrations",
        []() {
            nb::dict d;
            d["diffsoup_superlu_factor_f64"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluFactorF64));
            d["diffsoup_superlu_refactor_f64"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluRefactorF64));
            d["diffsoup_superlu_apply_f64"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluApplyF64));
            d["diffsoup_superlu_factor_c128"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluFactorC128));
            d["diffsoup_superlu_refactor_c128"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluRefactorC128));
            d["diffsoup_superlu_apply_c128"] =
                nb::capsule(reinterpret_cast<void*>(kSuperluApplyC128));
            return d;
        },
        "XLA FFI handler capsules by target name, for jax.ffi.register_ffi_target on the CPU.");
}
