// SuiteSparse KLU sparse LU on the host, exposed to JAX through XLA FFI handlers.
//
// Objects:
//   * an analysis per sparsity pattern: the compressed-column structure of
//     the operator, the map from the caller's coordinate entries to it, the
//     Dirichlet rows, the KLU control parameters and the symbolic
//     factorization from klu_analyze (block triangular form and fill-reducing
//     ordering);
//   * a slot per factor: the KLU numeric factorization, real or complex, and a
//     copy of the values it was built from;
//   * a site per lowered factor or refactor call: the slots that call writes
//     are keyed at the site by the context id the call carries at run time.
//     A factor call picks its slot by (context, slot token), so lanes
//     that carry distinct tokens get one slot each, every context gets its
//     own, and a repeated execution reuses them. A refactor call picks its
//     slot by (context, the handle it refactors), and refactors in place
//     when that handle is already one of its own slots, so a loop that
//     carries a factor keeps one slot.
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
// A slot that already holds a numeric factorization is refactored with
// klu_refactor, which keeps the pivot sequence and the storage of L and U.
// When the refactorization meets a zero pivot, or its reciprocal pivot ratio
// (klu_rcond) and condition estimate (klu_condest) say the kept pivot
// sequence has become inaccurate, the slot is factored afresh with
// klu_factor and its partial pivoting.
//
// Solves use klu_solve or klu_tsolve on the same factorization, with every
// right-hand side in one column-major block.
//
// Complex values: every handler has a complex128 twin (the targets ending in _c128) that runs
// the same steps with klu_z_factor, klu_z_refactor, klu_z_rcond, klu_z_condest, klu_z_solve and
// klu_z_tsolve on interleaved (real, imaginary) values. The symbolic analysis is the real one,
// since klu_analyze depends on the pattern alone. A transposed complex solve is the plain
// transpose (klu_z_tsolve with conj_solve = 0), the transpose that reverse-mode differentiation
// of a solve needs. A slot holds one kind of factorization; a call of the other kind factors it
// afresh, as a call with other values does. A solve whose values operand
// differs from the values its slot was last factored with refactors the slot
// from the operand first, so a solve never reads a factor built from other
// values.
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
// One mutex serialises every call into the library; the Python-facing
// functions release the GIL before taking it.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
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

#include "klu.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {

// Result codes shared with diff_soup.linear.SolveStatus.
constexpr int32_t kSuccess = 0;
constexpr int32_t kSingular = 1;
constexpr int32_t kNonfinite = 2;

// Below this reciprocal pivot ratio a refactorization is checked with
// klu_condest; above its inverse the slot is factored afresh. The value is
// the unit roundoff to the power 2/3.
const double kRcondFloor = std::pow(DBL_EPSILON, 2.0 / 3.0);

std::mutex& LibraryMutex() {
    static std::mutex mutex;
    return mutex;
}

struct Stats {
    int64_t factorizations = 0;      // klu_factor runs
    int64_t refactorizations = 0;    // klu_refactor runs whose result was kept
    int64_t refactor_fallbacks = 0;  // klu_refactor results replaced by klu_factor
    int64_t rebuilds = 0;            // factorizations started by a solve
    int64_t solves = 0;              // klu_solve and klu_tsolve calls
    int64_t nonfinite = 0;           // factor requests refused for non-finite values
    int64_t singular = 0;            // factor requests that ended singular
    int64_t released_context_errors = 0;  // calls refused because their context was released
};

// A slot's address at its site: (context id, slot token or held slot id).
using SiteKey = std::pair<int32_t, int64_t>;

struct Analysis {
    int64_t id = 0;
    int32_t n = 0;
    int64_t nnz_entries = 0;             // caller's coordinate entries
    std::vector<int32_t> colptr;         // n + 1
    std::vector<int32_t> rowind;         // compressed-column row indices
    std::vector<int64_t> entry_to_csc;   // coordinate entry -> compressed position
    std::vector<uint8_t> zeroed;         // coordinate entry lies in a Dirichlet row
    std::vector<int64_t> unit_diagonal;  // compressed positions that receive +1
    klu_common common{};
    klu_symbolic* symbolic = nullptr;

    Analysis() = default;
    Analysis(const Analysis&) = delete;
    Analysis& operator=(const Analysis&) = delete;
    ~Analysis() {
        if (symbolic != nullptr) {
            klu_free_symbolic(&symbolic, &common);
        }
    }
};

struct Slot {
    int32_t id = 0;
    int64_t site = 0;         // the key of the site that created the slot
    int64_t storage = 0;      // the storage key of a shared-storage slot, 0 for a site's slot
    // The XLA execution (run id) of the latest refactor of a shared-storage slot: a solve of the
    // same execution trusts the slot to hold its factor's matrix; any other compares values.
    // -1 once a solve that does not trust the slot has factored it again from its own values.
    int64_t trusted_run = -1;
    int32_t context = -1;     // the owning SolveContext's id
    SiteKey key{};            // the slot's address at its site
    std::shared_ptr<Analysis> analysis;
    klu_numeric* numeric = nullptr;
    bool has_values = false;  // `values` holds the operand of the last factor request
    bool complex = false;     // `numeric`, `values` and `csc` hold complex entries
    int32_t status = kSingular;
    std::vector<double> values;  // nnz entries, or 2 nnz interleaved for a complex slot
    std::vector<double> csc;     // the compressed matrix, interleaved for a complex slot

    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() { FreeNumeric(); }

    void FreeNumeric() {
        if (numeric != nullptr) {
            if (complex) {
                klu_z_free_numeric(&numeric, &analysis->common);
            } else {
                klu_free_numeric(&numeric, &analysis->common);
            }
            numeric = nullptr;
        }
    }
};

struct Site {
    int64_t id = 0;
    std::shared_ptr<Analysis> analysis;
    std::map<SiteKey, int32_t> slot_of_key;
};

struct Registry {
    std::unordered_map<int64_t, std::shared_ptr<Analysis>> analyses;
    std::unordered_map<int32_t, std::unique_ptr<Slot>> slots;
    std::unordered_map<int64_t, std::unique_ptr<Site>> sites;
    // Shared storage: the slot each (context id, device ordinal, analysis id, storage key,
    // empty-lane marker) names.
    std::map<std::tuple<int32_t, int32_t, int64_t, int64_t, int64_t>, int32_t> storage_slots;
    std::unordered_set<int32_t> released_contexts;
    int64_t next_analysis = 1;
    int32_t next_slot = 1;
    int64_t next_site = 1;  // ids of sites made by site_create
    Stats stats;
};

// The registry is never destroyed, so no KLU object is freed during process
// teardown after the library's own state is gone.
Registry& Reg() {
    static Registry* registry = new Registry();
    return *registry;
}

std::string KluStatusText(int status) {
    switch (status) {
        case KLU_OK:
            return "KLU_OK";
        case KLU_SINGULAR:
            return "KLU_SINGULAR";
        case KLU_OUT_OF_MEMORY:
            return "KLU_OUT_OF_MEMORY";
        case KLU_INVALID:
            return "KLU_INVALID";
        case KLU_TOO_LARGE:
            return "KLU_TOO_LARGE";
        default:
            return "status " + std::to_string(status);
    }
}

ffi::Error KluFailure(const char* routine, int status) {
    const auto code = status == KLU_OUT_OF_MEMORY ? ffi::ErrorCode::kResourceExhausted
                                                  : ffi::ErrorCode::kInternal;
    return ffi::Error(code, std::string("KLU ") + routine + " failed with " +
                                KluStatusText(status) +
                                (status == KLU_OUT_OF_MEMORY
                                     ? "; release unused SolveContexts or run on a host with "
                                       "more memory."
                                     : "."));
}

// Whether the refactorization in `slot` is accurate enough to keep: the
// reciprocal pivot ratio is above the floor, or the condition estimate is
// below its inverse.
bool RefactorAccurate(Slot& slot) {
    Analysis& a = *slot.analysis;
    const int rcond_ok = slot.complex ? klu_z_rcond(a.symbolic, slot.numeric, &a.common)
                                      : klu_rcond(a.symbolic, slot.numeric, &a.common);
    if (!rcond_ok) {
        return false;
    }
    if (a.common.rcond >= kRcondFloor) {
        return true;
    }
    const int condest_ok =
        slot.complex
            ? klu_z_condest(a.colptr.data(), slot.csc.data(), a.symbolic, slot.numeric, &a.common)
            : klu_condest(a.colptr.data(), slot.csc.data(), a.symbolic, slot.numeric, &a.common);
    if (!condest_ok) {
        return false;
    }
    return a.common.condest <= 1.0 / kRcondFloor;
}

// Factor `values` (the caller's coordinate entries, interleaved real and imaginary parts when
// `complex`) into `slot` and set its status. A slot that holds a factorization of the other
// kind is factored afresh. Caller holds the library mutex. Returns an error only for a KLU
// failure other than singularity (out of memory, integer overflow).
ffi::Error FactorSlot(Slot& slot, const double* values, bool complex) {
    Registry& reg = Reg();
    Analysis& a = *slot.analysis;
    const int64_t width = complex ? 2 : 1;
    const int64_t count = width * a.nnz_entries;
    if (slot.complex != complex) {
        slot.FreeNumeric();
        slot.complex = complex;
    }
    slot.values.assign(values, values + count);
    slot.has_values = true;

    for (int64_t k = 0; k < count; ++k) {
        if (!std::isfinite(values[k])) {
            slot.status = kNonfinite;
            ++reg.stats.nonfinite;
            return ffi::Error::Success();
        }
    }

    slot.csc.assign(static_cast<size_t>(width) * a.rowind.size(), 0.0);
    for (int64_t k = 0; k < a.nnz_entries; ++k) {
        if (!a.zeroed[k]) {
            for (int64_t part = 0; part < width; ++part) {
                slot.csc[width * a.entry_to_csc[k] + part] += values[width * k + part];
            }
        }
    }
    for (int64_t p : a.unit_diagonal) {
        slot.csc[width * p] += 1.0;
    }

    if (slot.numeric != nullptr) {
        const int ok =
            complex ? klu_z_refactor(a.colptr.data(), a.rowind.data(), slot.csc.data(),
                                     a.symbolic, slot.numeric, &a.common)
                    : klu_refactor(a.colptr.data(), a.rowind.data(), slot.csc.data(), a.symbolic,
                                   slot.numeric, &a.common);
        const int refactor_status = a.common.status;
        if (ok && refactor_status == KLU_OK && RefactorAccurate(slot)) {
            ++reg.stats.refactorizations;
            slot.status = kSuccess;
            return ffi::Error::Success();
        }
        if (refactor_status < KLU_OK) {
            slot.FreeNumeric();
            slot.status = kSingular;
            return KluFailure("klu_refactor", refactor_status);
        }
        ++reg.stats.refactor_fallbacks;
        slot.FreeNumeric();
    }

    slot.numeric =
        complex
            ? klu_z_factor(a.colptr.data(), a.rowind.data(), slot.csc.data(), a.symbolic,
                           &a.common)
            : klu_factor(a.colptr.data(), a.rowind.data(), slot.csc.data(), a.symbolic,
                         &a.common);
    ++reg.stats.factorizations;
    const int status = a.common.status;
    if (slot.numeric == nullptr) {
        slot.status = kSingular;
        if (status == KLU_SINGULAR) {
            ++reg.stats.singular;
            return ffi::Error::Success();
        }
        return KluFailure("klu_factor", status);
    }
    const bool rank_deficient = a.common.numerical_rank >= 0 && a.common.numerical_rank < a.n;
    if (status == KLU_SINGULAR || rank_deficient) {
        ++reg.stats.singular;
        slot.status = kSingular;
    } else {
        slot.status = kSuccess;
    }
    return ffi::Error::Success();
}

// Whether `slot` was last asked to factor exactly `values`, of the kind `complex`.
bool SameValues(const Slot& slot, const double* values, bool complex) {
    if (!slot.has_values || slot.complex != complex) {
        return false;
    }
    const size_t bytes = slot.values.size() * sizeof(double);
    return std::memcmp(slot.values.data(), values, bytes) == 0;
}

Slot* FindSlot(int32_t id) {
    auto& slots = Reg().slots;
    auto it = slots.find(id);
    return it == slots.end() ? nullptr : it->second.get();
}

Site* FindSite(int64_t id) {
    auto& sites = Reg().sites;
    auto it = sites.find(id);
    return it == sites.end() ? nullptr : it->second.get();
}

// The site of a lowered factor or refactor call: the site under `key`, created on the first
// call that carries the key, on the call's analysis. A site key is a deterministic hash of the
// lowered call, so two lowerings of one program share their sites. Caller holds the library
// mutex.
ffi::Error SiteFor(int64_t key, int64_t analysis_id, Site** out) {
    Registry& reg = Reg();
    Site* site = FindSite(key);
    if (site == nullptr) {
        auto found = reg.analyses.find(analysis_id);
        if (found == reg.analyses.end()) {
            return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                              "KLU analysis " + std::to_string(analysis_id) +
                                  " does not exist in this process (its Analysis was "
                                  "collected); analyze the pattern again and trace the call "
                                  "again.");
        }
        auto made = std::make_unique<Site>();
        made->id = key;
        made->analysis = found->second;
        site = made.get();
        reg.sites.emplace(key, std::move(made));
    }
    if (site->analysis->id != analysis_id) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU factor site " + std::to_string(key) +
                              " was created for analysis " +
                              std::to_string(site->analysis->id) + ", but the call names "
                              "analysis " + std::to_string(analysis_id) +
                              "; a site serves the analysis of the call that created it.");
    }
    *out = site;
    return ffi::Error::Success();
}

const char* kSlotsExhausted = "KLU slot identifiers are exhausted in this process.";

// The slot of `site` for `key` in `context`, created empty on first use.
// Caller holds the library mutex. Returns nullptr when slot identifiers are
// exhausted.
Slot* SiteSlot(Site& site, int32_t context, int64_t key) {
    Registry& reg = Reg();
    const SiteKey address{context, key};
    auto it = site.slot_of_key.find(address);
    if (it != site.slot_of_key.end()) {
        if (Slot* slot = FindSlot(it->second)) {
            return slot;
        }
    }
    if (reg.next_slot == std::numeric_limits<int32_t>::max()) {
        return nullptr;
    }
    auto slot = std::make_unique<Slot>();
    slot->id = reg.next_slot++;
    slot->site = site.id;
    slot->context = context;
    slot->key = address;
    slot->analysis = site.analysis;
    Slot* raw = slot.get();
    reg.slots.emplace(raw->id, std::move(slot));
    site.slot_of_key[address] = raw->id;
    return raw;
}

// The shared-storage slot of `storage` in `context` on device `device` for `marker` (the empty
// handle's slot id, which tells the lanes of a sequential batch apart), created empty on first
// use with `site` as its creator and kept in the registry's storage map instead of a site's. Caller holds the
// library mutex. Returns nullptr when slot identifiers are exhausted.
Slot* StorageSlot(Site& site, int32_t context, int32_t device, int64_t storage, int64_t marker) {
    Registry& reg = Reg();
    const auto key = std::make_tuple(context, device, site.analysis->id, storage, marker);
    auto it = reg.storage_slots.find(key);
    if (it != reg.storage_slots.end()) {
        if (Slot* slot = FindSlot(it->second)) {
            return slot;
        }
    }
    if (reg.next_slot == std::numeric_limits<int32_t>::max()) {
        return nullptr;
    }
    auto slot = std::make_unique<Slot>();
    slot->id = reg.next_slot++;
    slot->site = site.id;
    slot->storage = storage;
    slot->context = context;
    slot->key = SiteKey{context, std::numeric_limits<int64_t>::min()};
    slot->analysis = site.analysis;
    Slot* raw = slot.get();
    reg.slots.emplace(raw->id, std::move(slot));
    reg.storage_slots[key] = raw->id;
    return raw;
}

// Refuse a context that cannot own factors: none (a negative id) or one that
// was released. Caller holds the library mutex.
ffi::Error CheckContext(int32_t context) {
    Registry& reg = Reg();
    if (context < 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU keeps its factors in host storage that a SolveContext owns, but "
                          "this call carries no context (context id " +
                              std::to_string(context) +
                              "); pass ctx=diff_soup.linear.SolveContext() to factor().");
    }
    if (reg.released_contexts.count(context) != 0) {
        ++reg.stats.released_context_errors;
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "KLU: the SolveContext with context_id " + std::to_string(context) +
                              " was released, so the factors built in it no longer exist; build "
                              "the factor again in an open SolveContext.");
    }
    return ffi::Error::Success();
}

// The slot a handle (context, slot id) names, after checking that its
// context is open, the slot exists and belongs to that context. Caller holds
// the library mutex.
ffi::Error FindHeld(int32_t context, int32_t id, Slot** held) {
    if (auto err = CheckContext(context); err.failure()) {
        return err;
    }
    Slot* slot = FindSlot(id);
    if (slot == nullptr) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "KLU factor slot " + std::to_string(id) +
                              " does not exist in the SolveContext with context_id " +
                              std::to_string(context) +
                              "; pass a handle that diff_soup.linear.factor or refactor returned "
                              "in an open SolveContext.");
    }
    if (slot->context != context) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "KLU factor slot " + std::to_string(id) +
                              " belongs to the SolveContext with context_id " +
                              std::to_string(slot->context) + ", but the handle names context_id " +
                              std::to_string(context) +
                              "; pass the handle unchanged as factor or refactor returned it.");
    }
    *held = slot;
    return ffi::Error::Success();
}

// Bytes held by a numeric factorization: the L and U blocks, the diagonal of
// U, the row scaling, the permutations and column pointers, the off-diagonal
// blocks and the solve workspace.
int64_t NumericBytes(const klu_numeric* num, bool complex) {
    if (num == nullptr) {
        return 0;
    }
    const int64_t n = num->n;
    // L, U, the diagonal of U and the off-diagonal blocks hold complex entries in a complex
    // factorization; the row scaling is real either way.
    const int64_t entry = static_cast<int64_t>(sizeof(double)) * (complex ? 2 : 1);
    int64_t bytes = 0;
    // klu_factor allocates LUsize uninitialised and sets an entry only for a block it factors
    // with the kernel, which also sets the block's LUbx; a singleton block keeps its LUbx NULL
    // and its LUsize unset (KLU 2.3, klu_factor.c, factor2), so only kernel blocks count.
    for (int32_t block = 0; block < num->nblocks; ++block) {
        if (num->LUbx[block] != nullptr) {
            bytes += static_cast<int64_t>(num->LUsize[block]) * entry;
        }
    }
    bytes += n * entry;  // diagonal of U
    if (num->Rs != nullptr) {
        bytes += n * static_cast<int64_t>(sizeof(double));
    }
    bytes += 6 * n * static_cast<int64_t>(sizeof(int32_t));  // Pnum, Pinv, Lip, Uip, Llen, Ulen
    bytes += (n + 1) * static_cast<int64_t>(sizeof(int32_t));  // Offp
    bytes += static_cast<int64_t>(num->nzoff) * (static_cast<int64_t>(sizeof(int32_t)) + entry);
    bytes += static_cast<int64_t>(num->worksize);
    return bytes;
}

int64_t SlotBytes(const Slot& slot) {
    const int64_t copies = static_cast<int64_t>(slot.values.capacity() + slot.csc.capacity()) *
                           static_cast<int64_t>(sizeof(double));
    return copies + NumericBytes(slot.numeric, slot.complex);
}

// --------------------------------------------------------------------------
// XLA FFI handlers
// --------------------------------------------------------------------------

// The element type of a handler's values and right-hand sides: F64 or C128.
template <ffi::DataType T>
constexpr bool kIsComplex = T == ffi::DataType::C128;

// The values operand as interleaved doubles (one per entry, or two for a complex entry).
template <ffi::DataType T>
const double* ValueData(ffi::Buffer<T>& values) {
    return reinterpret_cast<const double*>(values.typed_data());
}

template <ffi::DataType T>
ffi::Error CheckValues(const Slot& slot, ffi::Buffer<T>& values) {
    if (static_cast<int64_t>(values.element_count()) != slot.analysis->nnz_entries) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU values operand has " + std::to_string(values.element_count()) +
                              " entries; the analysed pattern has " +
                              std::to_string(slot.analysis->nnz_entries) + ".");
    }
    return ffi::Error::Success();
}

// Refuse a slot built for another analysis than the call's (a handle passed
// with the parameters of a different sparsity pattern).
ffi::Error CheckAnalysis(const Slot& slot, int64_t analysis_id) {
    if (slot.analysis->id != analysis_id) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU: the handle names a factor slot built for a different sparsity "
                          "pattern (slot " +
                              std::to_string(slot.id) + " belongs to analysis " +
                              std::to_string(slot.analysis->id) + "; this call's analysis is " +
                              std::to_string(analysis_id) +
                              "); solve and refactor a factor with the Analysis it was built "
                              "with.");
    }
    return ffi::Error::Success();
}

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
    std::lock_guard<std::mutex> lock(LibraryMutex());
    if (auto err = CheckContext(context); err.failure()) {
        return err;
    }
    Site* site = nullptr;
    if (auto err = SiteFor(site_id, analysis_id, &site); err.failure()) {
        return err;
    }
    // A factor of shared storage (the fill of an empty factor that names it) goes to the
    // storage's slot, keyed by the empty handle's slot id; any other to the site's slot.
    Slot* slot =
        storage != 0 ? StorageSlot(*site, context, device, storage, token)
                     : SiteSlot(*site, context, token);
    if (slot == nullptr) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, kSlotsExhausted);
    }
    if (auto err = CheckValues(*slot, values); err.failure()) {
        return err;
    }
    if (auto err = FactorSlot(*slot, ValueData(values), kIsComplex<T>); err.failure()) {
        return err;
    }
    status->typed_data()[0] = slot->status;
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
    std::lock_guard<std::mutex> lock(LibraryMutex());
    Slot* held = nullptr;
    if (auto err = FindHeld(context, held_id, &held); err.failure()) {
        return err;
    }
    if (auto err = CheckAnalysis(*held, analysis_id); err.failure()) {
        return err;
    }
    handle_context->typed_data()[0] = context;
    // A factor of shared storage makes the storage hold the values it keeps (the values operand
    // carries them): with a false predicate it is refactored only when the storage holds other
    // values, for instance at the entry of a replayed segment.
    if (!refactor && (storage == 0 || SameValues(*held, ValueData(values), kIsComplex<T>))) {
        if (storage != 0) held->trusted_run = run.run_id;
        handle_slot->typed_data()[0] = held_id;
        status->typed_data()[0] = held->status;
        return ffi::Error::Success();
    }
    Site* site = nullptr;
    if (auto err = SiteFor(site_id, analysis_id, &site); err.failure()) {
        return err;
    }
    // In place when the held factor is already this site's or names shared storage; otherwise
    // the site's slot for that held factor, so other holders of it keep it.
    Slot* slot = held->site == site_id || storage != 0 ? held : SiteSlot(*site, context, held_id);
    if (slot == nullptr) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted, kSlotsExhausted);
    }
    if (auto err = CheckValues(*slot, values); err.failure()) {
        return err;
    }
    if (auto err = FactorSlot(*slot, ValueData(values), kIsComplex<T>); err.failure()) {
        return err;
    }
    status->typed_data()[0] = slot->status;
    if (storage != 0) slot->trusted_run = run.run_id;
    handle_slot->typed_data()[0] = slot->id;
    return ffi::Error::Success();
}

// A complex solve takes values, right-hand sides and solution in complex128; a real factor
// with a complex right-hand side is solved by the lowering as one real block of the real and
// imaginary parts, so no handler mixes the kinds.
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
    std::lock_guard<std::mutex> lock(LibraryMutex());
    Registry& reg = Reg();
    Slot* slot = nullptr;
    if (auto err = FindHeld(context, id, &slot); err.failure()) {
        return err;
    }
    if (auto err = CheckAnalysis(*slot, analysis_id); err.failure()) {
        return err;
    }
    if (auto err = CheckValues(*slot, values); err.failure()) {
        return err;
    }
    const int64_t n = slot->analysis->n;
    const int64_t total = static_cast<int64_t>(b.element_count());
    if (n == 0 || total % n != 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU right-hand side has " + std::to_string(total) +
                              " entries, which is not a whole number of columns of length " +
                              std::to_string(n) + ".");
    }
    const int64_t nrhs = total / n;
    if (nrhs > std::numeric_limits<int32_t>::max()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "KLU solves at most 2**31 - 1 right-hand sides in one block; got " +
                              std::to_string(nrhs) + ". Split the block.");
    }

    // A solve of a factor that names shared storage (storage != 0) uses the storage as it
    // stands when a refactor of it ran earlier in this execution (that refactor made it hold
    // the matrix of the factor it returned); any other solve refactors a slot that holds other
    // values than the ones it carries. Such a refactor (a derivative solve of an older factor
    // runs after a later refactor of the storage) leaves the slot holding this solve's matrix,
    // so the slot stops being trusted: a later solve of this execution compares values again.
    constexpr bool complex = kIsComplex<T>;
    // A slot of the other kind (a shared storage first filled by a factor of the other kind)
    // never counts as holding these values, so it is factored afresh like a slot of other values.
    const bool trusted = storage != 0 && slot->trusted_run == run.run_id && slot->complex == complex;
    if (!trusted && !SameValues(*slot, ValueData(values), complex)) {
        slot->trusted_run = -1;
        if (auto err = FactorSlot(*slot, ValueData(values), complex); err.failure()) {
            return err;
        }
        ++reg.stats.rebuilds;
    }

    // Interleaved doubles: one per entry, or the real and imaginary parts of a complex entry.
    const int64_t width = complex ? 2 : 1;
    double* out = reinterpret_cast<double*>(x->typed_data());
    status->typed_data()[0] = slot->status;
    if (slot->status != kSuccess) {
        std::fill(out, out + width * total, std::numeric_limits<double>::quiet_NaN());
        return ffi::Error::Success();
    }
    if (total == 0) {
        return ffi::Error::Success();
    }
    std::memcpy(out, b.typed_data(), static_cast<size_t>(width * total) * sizeof(double));

    Analysis& a = *slot->analysis;
    const auto ldim = static_cast<int32_t>(n);
    const auto cols = static_cast<int32_t>(nrhs);
    int ok = 0;
    if constexpr (complex) {
        // conj_solve = 0: the plain transpose A^T, never the conjugate transpose.
        ok = transpose ? klu_z_tsolve(a.symbolic, slot->numeric, ldim, cols, out, 0, &a.common)
                       : klu_z_solve(a.symbolic, slot->numeric, ldim, cols, out, &a.common);
    } else {
        ok = transpose ? klu_tsolve(a.symbolic, slot->numeric, ldim, cols, out, &a.common)
                       : klu_solve(a.symbolic, slot->numeric, ldim, cols, out, &a.common);
    }
    ++reg.stats.solves;
    if (!ok) {
        const char* routine = complex ? (transpose ? "klu_z_tsolve" : "klu_z_solve")
                                      : (transpose ? "klu_tsolve" : "klu_solve");
        return KluFailure(routine, a.common.status);
    }
    return ffi::Error::Success();
}

}  // namespace

// One binding per element type: the values (and right-hand sides and solution) are F64 for the
// real handlers and C128 for the complex ones; everything else is the same.
#define DIFFSOUP_KLU_HANDLERS(SUFFIX, TYPE)                                                       \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                                \
        kKluFactor##SUFFIX, FactorImpl<TYPE>,                                                     \
        ffi::Ffi::Bind()                                                                          \
            .Ctx<ffi::RunId>()            /* the XLA execution */                                 \
            .Ctx<ffi::DeviceOrdinal>()    /* the device that runs the call: keys shared storage */\
            .Arg<ffi::Token>()            /* XLA token in (fresh per call) */                     \
            .Arg<ffi::Buffer<TYPE>>()     /* values */                                            \
            .Arg<ffi::Buffer<ffi::S32>>() /* slot key (call-site token) */                        \
            .Arg<ffi::Buffer<ffi::S32>>() /* context id */                                        \
            .Arg<ffi::Buffer<ffi::S32>>() /* order word in (read by no one: it orders the call) */\
            .Ret<ffi::Token>()            /* XLA token out */                                     \
            .Ret<ffi::Buffer<ffi::S32>>() /* handle: context id */                                \
            .Ret<ffi::Buffer<ffi::S32>>() /* handle: slot */                                      \
            .Ret<ffi::Buffer<ffi::S32>>() /* status */                                            \
            .Attr<int64_t>("site")                                                                \
            .Attr<int64_t>("analysis_id")                                                         \
            .Attr<int64_t>("storage"));                                                           \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                                \
        kKluRefactor##SUFFIX, RefactorImpl<TYPE>,                                                 \
        ffi::Ffi::Bind()                                                                          \
            .Ctx<ffi::RunId>()             /* the XLA execution */                                \
            .Arg<ffi::Token>()             /* XLA token in (fresh per call) */                    \
            .Arg<ffi::Buffer<TYPE>>()      /* values */                                           \
            .Arg<ffi::Buffer<ffi::PRED>>() /* predicate */                                        \
            .Arg<ffi::Buffer<ffi::S32>>()  /* held handle: context id */                          \
            .Arg<ffi::Buffer<ffi::S32>>()  /* held handle: slot */                                \
            .Arg<ffi::Buffer<ffi::S32>>()  /* order word in (read by no one) */                   \
            .Ret<ffi::Token>()             /* XLA token out */                                    \
            .Ret<ffi::Buffer<ffi::S32>>()  /* handle: context id */                               \
            .Ret<ffi::Buffer<ffi::S32>>()  /* handle: slot */                                     \
            .Ret<ffi::Buffer<ffi::S32>>()  /* status */                                           \
            .Attr<int64_t>("site")                                                                \
            .Attr<int64_t>("analysis_id")                                                         \
            .Attr<int64_t>("storage"));                                                           \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                                \
        kKluApply##SUFFIX, ApplyImpl<TYPE>,                                                       \
        ffi::Ffi::Bind()                                                                          \
            .Ctx<ffi::RunId>()            /* the XLA execution */                                 \
            .Arg<ffi::Token>()            /* XLA token in (fresh per call) */                     \
            .Arg<ffi::Buffer<TYPE>>()     /* values */                                            \
            .Arg<ffi::Buffer<ffi::S32>>() /* handle: context id */                                \
            .Arg<ffi::Buffer<ffi::S32>>() /* handle: slot */                                      \
            .Arg<ffi::Buffer<TYPE>>()     /* right-hand sides */                                  \
            .Arg<ffi::Buffer<ffi::S32>>() /* order word in (read by no one) */                    \
            .Ret<ffi::Token>()            /* XLA token out */                                     \
            .Ret<ffi::Buffer<TYPE>>()     /* solution */                                          \
            .Ret<ffi::Buffer<ffi::S32>>() /* status */                                            \
            .Attr<bool>("transpose")                                                              \
            .Attr<int64_t>("analysis_id")                                                         \
            .Attr<int64_t>("storage"))

DIFFSOUP_KLU_HANDLERS(F64, ffi::F64);
DIFFSOUP_KLU_HANDLERS(C128, ffi::C128);

#undef DIFFSOUP_KLU_HANDLERS

// --------------------------------------------------------------------------
// Python module
// --------------------------------------------------------------------------

namespace {

using IndexArray = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using MaskArray = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

int64_t AnalysisCreate(int64_t n, IndexArray rows, IndexArray cols, MaskArray zeroed,
                       IndexArray unit_rows, int64_t ordering, bool btf, int64_t scale,
                       double pivot_tolerance) {
    const int64_t nnz = static_cast<int64_t>(rows.shape(0));
    if (static_cast<int64_t>(cols.shape(0)) != nnz ||
        static_cast<int64_t>(zeroed.shape(0)) != nnz) {
        throw std::invalid_argument(
            "KLU analysis needs rows, cols and the Dirichlet mask of equal length.");
    }
    if (n <= 0 || n > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("KLU analysis needs 0 < n <= 2**31 - 1; got n = " +
                                    std::to_string(n) + ".");
    }
    if (ordering != 0 && ordering != 1) {
        throw std::invalid_argument("KLU ordering code must be 0 (AMD) or 1 (COLAMD); got " +
                                    std::to_string(ordering) + ".");
    }
    if (scale < 0 || scale > 2) {
        throw std::invalid_argument(
            "KLU row scaling code must be 0 (none), 1 (sum) or 2 (max); got " +
            std::to_string(scale) + ".");
    }
    if (!(pivot_tolerance > 0.0 && pivot_tolerance <= 1.0)) {
        throw std::invalid_argument("KLU pivot tolerance must lie in (0, 1]; got " +
                                    std::to_string(pivot_tolerance) + ".");
    }
    const int32_t* r = rows.data();
    const int32_t* c = cols.data();
    for (int64_t k = 0; k < nnz; ++k) {
        if (r[k] < 0 || r[k] >= n || c[k] < 0 || c[k] >= n) {
            throw std::invalid_argument("KLU analysis: pattern entry " + std::to_string(k) +
                                        " lies outside the " + std::to_string(n) + " x " +
                                        std::to_string(n) + " operator.");
        }
    }

    auto a = std::make_shared<Analysis>();
    a->n = static_cast<int32_t>(n);
    a->nnz_entries = nnz;
    a->zeroed.assign(zeroed.data(), zeroed.data() + nnz);

    {
        nb::gil_scoped_release release;
        // Unique (row, col) pairs in column-major order.
        std::vector<int64_t> order(static_cast<size_t>(nnz));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
            return c[x] != c[y] ? c[x] < c[y] : r[x] < r[y];
        });
        a->entry_to_csc.assign(static_cast<size_t>(nnz), 0);
        a->colptr.assign(static_cast<size_t>(n) + 1, 0);
        a->rowind.clear();
        a->rowind.reserve(static_cast<size_t>(nnz));
        int64_t prev_r = -1, prev_c = -1;
        for (int64_t k : order) {
            if (r[k] != prev_r || c[k] != prev_c) {
                if (a->rowind.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::invalid_argument(
                        "KLU analysis: the pattern has more than 2**31 - 1 distinct entries.");
                }
                a->rowind.push_back(r[k]);
                a->colptr[static_cast<size_t>(c[k]) + 1] += 1;
                prev_r = r[k];
                prev_c = c[k];
            }
            a->entry_to_csc[k] = static_cast<int64_t>(a->rowind.size()) - 1;
        }
        for (int64_t j = 0; j < n; ++j) {
            a->colptr[j + 1] += a->colptr[j];
        }
        // Compressed position of each Dirichlet row's diagonal; the unit is
        // added once per row whatever the duplicates.
        const int32_t* u = unit_rows.data();
        for (size_t i = 0; i < unit_rows.shape(0); ++i) {
            const int32_t row = u[i];
            if (row < 0 || row >= n) {
                throw std::invalid_argument("KLU analysis: Dirichlet row " +
                                            std::to_string(row) + " lies outside the operator.");
            }
            const int32_t* begin = a->rowind.data() + a->colptr[row];
            const int32_t* end = a->rowind.data() + a->colptr[row + 1];
            const int32_t* hit = std::lower_bound(begin, end, row);
            if (hit == end || *hit != row) {
                throw std::invalid_argument("KLU analysis: Dirichlet row " +
                                            std::to_string(row) +
                                            " has no diagonal entry in the pattern.");
            }
            const int64_t position = static_cast<int64_t>(hit - a->rowind.data());
            if (std::find(a->unit_diagonal.begin(), a->unit_diagonal.end(), position) ==
                a->unit_diagonal.end()) {
                a->unit_diagonal.push_back(position);
            }
        }

        std::lock_guard<std::mutex> lock(LibraryMutex());
        klu_defaults(&a->common);
        a->common.ordering = static_cast<int>(ordering);
        a->common.btf = btf ? 1 : 0;
        a->common.scale = static_cast<int>(scale);
        a->common.tol = pivot_tolerance;
        a->common.halt_if_singular = 1;
        a->symbolic = klu_analyze(a->n, a->colptr.data(), a->rowind.data(), &a->common);
        if (a->symbolic == nullptr) {
            throw std::invalid_argument("KLU klu_analyze rejected the pattern with " +
                                        KluStatusText(a->common.status) + ".");
        }
        Registry& reg = Reg();
        a->id = reg.next_analysis++;
        reg.analyses[a->id] = a;
    }
    return a->id;
}

void AnalysisRelease(int64_t id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(LibraryMutex());
    Reg().analyses.erase(id);  // sites and slots keep their analysis alive
}

int64_t SiteCreate(int64_t analysis_id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(LibraryMutex());
    Registry& reg = Reg();
    auto it = reg.analyses.find(analysis_id);
    if (it == reg.analyses.end()) {
        throw std::invalid_argument("KLU analysis " + std::to_string(analysis_id) +
                                    " does not exist; analyse the pattern again.");
    }
    while (reg.sites.count(reg.next_site) != 0) ++reg.next_site;
    auto site = std::make_unique<Site>();
    site->id = reg.next_site++;
    site->analysis = it->second;
    const int64_t id = site->id;
    reg.sites.emplace(id, std::move(site));
    return id;
}

void ContextRelease(int64_t context) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(LibraryMutex());
    Registry& reg = Reg();
    for (auto it = reg.slots.begin(); it != reg.slots.end();) {
        const Slot& slot = *it->second;
        if (slot.context != context) {
            ++it;
            continue;
        }
        if (Site* site = FindSite(slot.site); site != nullptr && slot.storage == 0) {
            site->slot_of_key.erase(slot.key);
        }
        it = reg.slots.erase(it);  // the slot frees its numeric factorization
    }
    for (auto it = reg.storage_slots.begin(); it != reg.storage_slots.end();) {
        it = std::get<0>(it->first) == context ? reg.storage_slots.erase(it) : std::next(it);
    }
    reg.released_contexts.insert(static_cast<int32_t>(context));
}

int64_t ResidentBytes(int64_t context) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(LibraryMutex());
    int64_t total = 0;
    for (const auto& entry : Reg().slots) {
        if (entry.second->context == context) {
            total += SlotBytes(*entry.second);
        }
    }
    return total;
}

}  // namespace

NB_MODULE(_native_klu, m) {
    m.doc() = "SuiteSparse KLU sparse LU factorization exposed through XLA FFI handlers.";

    m.def("analysis_create", &AnalysisCreate, nb::arg("n"), nb::arg("rows"), nb::arg("cols"),
          nb::arg("zeroed"), nb::arg("unit_rows"), nb::arg("ordering"), nb::arg("btf"),
          nb::arg("scale"), nb::arg("pivot_tolerance"),
          "Analyse a sparsity pattern: compressed-column structure, Dirichlet rows and the "
          "KLU symbolic factorization. Returns the analysis id.");
    m.def("analysis_release", &AnalysisRelease, nb::arg("analysis_id"),
          "Drop the registry's reference to an analysis; sites and slots that use it keep it "
          "alive.");
    m.def("site_create", &SiteCreate, nb::arg("analysis_id"),
          "Create a factor site on an analysis with an id of its own (a lowered call names its "
          "site by a key computed from the call, and the site is created on the first call "
          "that carries it); its slots are keyed by the context id each execution carries. "
          "Returns the site id.");
    m.def("context_release", &ContextRelease, nb::arg("context_id"),
          "Free every slot of the given SolveContext id and record the id as released, so a "
          "later call that carries it fails.");
    m.def("resident_bytes", &ResidentBytes, nb::arg("context_id"),
          "Bytes held by the slots of the given SolveContext id: value copies and the numeric "
          "factorizations.");
    m.def(
        "stats",
        []() {
            Stats s;
            int64_t live_slots = 0;
            int64_t live_analyses = 0;
            int64_t live_sites = 0;
            int64_t live_lu_entries = 0;
            {
                nb::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(LibraryMutex());
                s = Reg().stats;
                live_slots = static_cast<int64_t>(Reg().slots.size());
                live_analyses = static_cast<int64_t>(Reg().analyses.size());
                live_sites = static_cast<int64_t>(Reg().sites.size());
                for (const auto& entry : Reg().slots) {
                    if (const klu_numeric* num = entry.second->numeric) {
                        live_lu_entries += static_cast<int64_t>(num->lnz) + num->unz;
                    }
                }
            }
            nb::dict d;
            d["factorizations"] = s.factorizations;
            d["refactorizations"] = s.refactorizations;
            d["refactor_fallbacks"] = s.refactor_fallbacks;
            d["rebuilds"] = s.rebuilds;
            d["solves"] = s.solves;
            d["nonfinite"] = s.nonfinite;
            d["singular"] = s.singular;
            d["released_context_errors"] = s.released_context_errors;
            d["live_slots"] = live_slots;
            d["live_analyses"] = live_analyses;
            d["live_sites"] = live_sites;
            d["live_lu_entries"] = live_lu_entries;
            return d;
        },
        "Process-wide counters of factorizations (klu_factor), refactorizations "
        "(klu_refactor kept), refactorizations replaced by a factorization, rebuilds started "
        "by a solve, solves, refused and singular requests, calls refused for a released "
        "context, live objects, and the entries of L and U (diagonals included) over the live "
        "numeric factorizations.");
    m.def(
        "reset_stats",
        []() {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(LibraryMutex());
            Reg().stats = Stats{};
        },
        "Zero the counters reported by stats().");
    m.def(
        "registrations",
        []() {
            nb::dict d;
            d["diffsoup_klu_factor_f64"] = nb::capsule(reinterpret_cast<void*>(kKluFactorF64));
            d["diffsoup_klu_refactor_f64"] =
                nb::capsule(reinterpret_cast<void*>(kKluRefactorF64));
            d["diffsoup_klu_apply_f64"] = nb::capsule(reinterpret_cast<void*>(kKluApplyF64));
            d["diffsoup_klu_factor_c128"] = nb::capsule(reinterpret_cast<void*>(kKluFactorC128));
            d["diffsoup_klu_refactor_c128"] =
                nb::capsule(reinterpret_cast<void*>(kKluRefactorC128));
            d["diffsoup_klu_apply_c128"] = nb::capsule(reinterpret_cast<void*>(kKluApplyC128));
            return d;
        },
        "XLA FFI handler capsules by target name, for jax.ffi.register_ffi_target on the CPU.");
}
