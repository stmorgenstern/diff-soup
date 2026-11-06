// The slot registry of diff-soup's native linear backends: the structures and functions that
// the KLU (klu_ffi.cpp), SuperLU_MT (superlu_solver.cpp) and cuDSS (contract_handlers.inc)
// extensions share, written once and instantiated with each extension's own analysis, slot,
// site and counter types.
//
// The registry of an extension holds, under one mutex:
//   * the analyses by id;
//   * the slots by id: a slot holds one factor, belongs to one context (its handle is the pair
//     (context id, slot id)) and records the site that created it;
//   * the sites by 64-bit key: a site is a lowered factor or refactor call, created on the
//     first call that carries its key, and keys its slots by (context id, slot token or held
//     slot id);
//   * the shared-storage slots, keyed by a tuple that starts with the context id;
//   * the released contexts, whose calls are refused.
//
// The host backends (KLU, SuperLU_MT) create their slots with SiteSlot and StorageSlot below.
// cuDSS creates its own (a device handle per slot, a range of ids per batch slot, a precision
// and lane count in its storage key) and takes the rest.
//
// Each extension's registry derives from RegistryBase and names its backend and where its
// factors live:
//     struct Registry : slot_registry::RegistryBase<Analysis, Slot, Site, Stats, StorageKey> {
//         static constexpr const char* kName = "KLU";
//         static constexpr const char* kStorage = "host storage";
//     };
// Its slots derive from SlotBase, its sites are or derive from SiteBase, and its Stats has the
// counter released_context_errors. Every function that takes a registry expects the caller to
// hold its mutex, except the Python-facing ones (AnalysisRelease, SiteCreate, ContextRelease,
// ResidentBytes), which release the GIL and take it.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace slot_registry {

namespace ffi = xla::ffi;
namespace nb = nanobind;

// A slot's address at its site: (context id, slot token or held slot id).
using SiteKey = std::pair<int32_t, int64_t>;

// The shared-storage key of a host slot: (context id, device ordinal, analysis id, storage key,
// empty-lane marker). The device keeps apart the slots of the devices that run one program;
// the marker (the empty handle's slot id) keeps apart the lanes of a sequential batch.
using HostStorageKey = std::tuple<int32_t, int32_t, int64_t, int64_t, int64_t>;

// The registry's fields of a slot.
template <class AnalysisT>
struct SlotBase {
    int32_t id = 0;        // the slot's (first) id
    int64_t site = 0;      // the key of the site that created the slot
    int64_t storage = 0;   // the storage key of a shared-storage slot, 0 for a site's slot
    // The XLA execution (run id) of the latest refactor of a shared-storage slot: a solve of the
    // same execution trusts the slot to hold its factor's matrix. -1 once a solve that does not
    // trust the slot has factored it again from its own values.
    int64_t trusted_run = -1;
    int32_t context = -1;  // the owning SolveContext's id
    SiteKey key{};         // the slot's address at its site (unused for shared storage)
    std::shared_ptr<AnalysisT> analysis;
};

template <class AnalysisT>
struct SiteBase {
    int64_t id = 0;
    std::shared_ptr<AnalysisT> analysis;
    std::map<SiteKey, int32_t> slot_of_key;
};

template <class AnalysisT, class SlotT, class SiteT, class StatsT, class StorageKeyT>
struct RegistryBase {
    using AnalysisType = AnalysisT;
    using SlotType = SlotT;
    using SiteType = SiteT;

    std::mutex mutex;  // serialises every call into the extension's library
    std::unordered_map<int64_t, std::shared_ptr<AnalysisT>> analyses;
    std::map<int32_t, std::unique_ptr<SlotT>> slots;  // by the slot's (first) id
    std::unordered_map<int64_t, std::unique_ptr<SiteT>> sites;
    std::map<StorageKeyT, int32_t> storage_slots;  // shared storage: the slot each key names
    std::unordered_set<int32_t> released_contexts;
    int64_t next_analysis = 1;
    int32_t next_slot = 1;
    int64_t next_site = 1;  // ids of sites made by SiteCreate
    StatsT stats;
};

// The registry of type R. It is never destroyed, so no library object is freed during process
// teardown after the library's own state (or the CUDA context) is gone.
template <class R>
R& Instance() {
    static R* registry = new R();
    return *registry;
}

// --------------------------------------------------------------------------
// Lookups and checks
// --------------------------------------------------------------------------

// The slot whose (first) id is `id`, or null.
template <class R>
typename R::SlotType* FindSlot(R& reg, int32_t id) {
    auto it = reg.slots.find(id);
    return it == reg.slots.end() ? nullptr : it->second.get();
}

template <class R>
typename R::SiteType* FindSite(R& reg, int64_t id) {
    auto it = reg.sites.find(id);
    return it == reg.sites.end() ? nullptr : it->second.get();
}

// Refuse a context that cannot own factors: none (a negative id) or one that was released.
template <class R>
ffi::Error CheckContext(R& reg, int32_t context) {
    if (context < 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string(R::kName) + " keeps its factors in " + R::kStorage +
                              " that a SolveContext owns, but this call carries no context "
                              "(context id " + std::to_string(context) +
                              "); pass ctx=diff_soup.linear.SolveContext() to factor().");
    }
    if (reg.released_contexts.count(context) != 0) {
        ++reg.stats.released_context_errors;
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          std::string(R::kName) + ": the SolveContext with context_id " +
                              std::to_string(context) +
                              " was released, so the factors built in it no longer exist; "
                              "build the factor again in an open SolveContext.");
    }
    return ffi::Error::Success();
}

// The slot `slot` that a handle (context, id) names, after checking that the context is open,
// the slot exists (is not null) and belongs to that context.
template <class R>
ffi::Error CheckHeld(R& reg, int32_t context, int32_t id, typename R::SlotType* slot,
                     typename R::SlotType** held) {
    if (auto err = CheckContext(reg, context); err.failure()) {
        return err;
    }
    if (slot == nullptr) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          std::string(R::kName) + " factor slot " + std::to_string(id) +
                              " does not exist in the SolveContext with context_id " +
                              std::to_string(context) +
                              "; pass a handle that diff_soup.linear.factor or refactor "
                              "returned in an open SolveContext of this process.");
    }
    if (slot->context != context) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          std::string(R::kName) + " factor slot " + std::to_string(id) +
                              " belongs to the SolveContext with context_id " +
                              std::to_string(slot->context) +
                              ", but the handle names context_id " + std::to_string(context) +
                              "; pass the handle unchanged as factor or refactor returned it.");
    }
    *held = slot;
    return ffi::Error::Success();
}

// The slot whose id a handle (context, id) names, checked as CheckHeld checks it.
template <class R>
ffi::Error FindHeld(R& reg, int32_t context, int32_t id, typename R::SlotType** held) {
    return CheckHeld(reg, context, id, FindSlot(reg, id), held);
}

// Refuse a slot built for another analysis than the call's (a handle passed with the parameters
// of a different sparsity pattern).
template <class R>
ffi::Error CheckAnalysis(const R&, const typename R::SlotType& slot, int64_t analysis_id) {
    if (slot.analysis->id != analysis_id) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string(R::kName) +
                              ": the handle names a factor slot built for a different sparsity "
                              "pattern (slot " + std::to_string(slot.id) +
                              " belongs to analysis " + std::to_string(slot.analysis->id) +
                              "; this call's analysis is " + std::to_string(analysis_id) +
                              "); solve and refactor a factor with the Analysis it was built "
                              "with.");
    }
    return ffi::Error::Success();
}

// The site of a lowered factor or refactor call: the site under `key`, created on the first call
// that carries the key, on the call's analysis. A site key is a deterministic hash of the lowered
// call, so two lowerings of one program share their sites, or an id SiteCreate made.
template <class R>
ffi::Error SiteFor(R& reg, int64_t key, int64_t analysis_id, typename R::SiteType** out) {
    using SiteT = typename R::SiteType;
    SiteT* site = FindSite(reg, key);
    if (site == nullptr) {
        auto found = reg.analyses.find(analysis_id);
        if (found == reg.analyses.end()) {
            return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                              std::string(R::kName) + " analysis " +
                                  std::to_string(analysis_id) +
                                  " does not exist in this process (its Analysis was "
                                  "collected); analyze the pattern again and trace the call "
                                  "again.");
        }
        auto made = std::make_unique<SiteT>();
        made->id = key;
        made->analysis = found->second;
        site = made.get();
        reg.sites.emplace(key, std::move(made));
    }
    if (site->analysis->id != analysis_id) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string(R::kName) + " factor site " + std::to_string(key) +
                              " was created for analysis " +
                              std::to_string(site->analysis->id) +
                              ", but the call names analysis " + std::to_string(analysis_id) +
                              "; a site serves the analysis of the call that created it.");
    }
    *out = site;
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Slots
// --------------------------------------------------------------------------

// Refuse a slot of `lanes` consecutive ids when fewer are left in the process.
template <class R>
ffi::Error CheckSlotIds(const R& reg, int64_t lanes) {
    if (lanes < 1 || reg.next_slot > std::numeric_limits<int32_t>::max() - lanes) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          std::string(R::kName) +
                              " slot identifiers are exhausted in this process.");
    }
    return ffi::Error::Success();
}

// Give `slot` the next `lanes` ids, after CheckSlotIds, and keep it in the registry.
template <class R>
typename R::SlotType* AddSlot(R& reg, std::unique_ptr<typename R::SlotType> slot,
                              int64_t lanes) {
    slot->id = reg.next_slot;
    reg.next_slot += static_cast<int32_t>(lanes);
    auto* raw = slot.get();
    reg.slots.emplace(raw->id, std::move(slot));
    return raw;
}

// The host slot of `site` for `key` in `context`, created empty on first use.
template <class R>
ffi::Error SiteSlot(R& reg, typename R::SiteType& site, int32_t context, int64_t key,
                    typename R::SlotType** out) {
    using SlotT = typename R::SlotType;
    const SiteKey address{context, key};
    auto it = site.slot_of_key.find(address);
    if (it != site.slot_of_key.end()) {
        if (SlotT* slot = FindSlot(reg, it->second)) {
            *out = slot;
            return ffi::Error::Success();
        }
    }
    if (auto err = CheckSlotIds(reg, 1); err.failure()) {
        return err;
    }
    auto slot = std::make_unique<SlotT>();
    slot->site = site.id;
    slot->context = context;
    slot->key = address;
    slot->analysis = site.analysis;
    SlotT* raw = AddSlot(reg, std::move(slot), 1);
    site.slot_of_key[address] = raw->id;
    *out = raw;
    return ffi::Error::Success();
}

// The host shared-storage slot of `storage` in `context` on device `device` for `marker` (the
// empty handle's slot id, which tells the lanes of a sequential batch apart), created empty on
// first use with `site` as its creator and kept in the registry's storage map instead of a
// site's.
template <class R>
ffi::Error StorageSlot(R& reg, typename R::SiteType& site, int32_t context, int32_t device,
                       int64_t storage, int64_t marker, typename R::SlotType** out) {
    using SlotT = typename R::SlotType;
    const HostStorageKey key{context, device, site.analysis->id, storage, marker};
    auto it = reg.storage_slots.find(key);
    if (it != reg.storage_slots.end()) {
        if (SlotT* slot = FindSlot(reg, it->second)) {
            *out = slot;
            return ffi::Error::Success();
        }
    }
    if (auto err = CheckSlotIds(reg, 1); err.failure()) {
        return err;
    }
    auto slot = std::make_unique<SlotT>();
    slot->site = site.id;
    slot->storage = storage;
    slot->context = context;
    slot->key = SiteKey{context, std::numeric_limits<int64_t>::min()};
    slot->analysis = site.analysis;
    SlotT* raw = AddSlot(reg, std::move(slot), 1);
    reg.storage_slots[key] = raw->id;
    *out = raw;
    return ffi::Error::Success();
}

// Free every slot of `context` (calling `on_slot` on each just before it is destroyed), forget
// its site addresses and shared-storage keys, and record the context as released.
template <class R, class OnSlot>
void ReleaseContextSlots(R& reg, int32_t context, OnSlot&& on_slot) {
    for (auto it = reg.slots.begin(); it != reg.slots.end();) {
        auto& slot = *it->second;
        if (slot.context != context) {
            ++it;
            continue;
        }
        if (auto* site = FindSite(reg, slot.site); site != nullptr && slot.storage == 0) {
            site->slot_of_key.erase(slot.key);
        }
        on_slot(slot);
        it = reg.slots.erase(it);  // the slot frees its factorization
    }
    for (auto it = reg.storage_slots.begin(); it != reg.storage_slots.end();) {
        it = std::get<0>(it->first) == context ? reg.storage_slots.erase(it) : std::next(it);
    }
    reg.released_contexts.insert(context);
}

// --------------------------------------------------------------------------
// Host values
// --------------------------------------------------------------------------

// Whether a buffer of element type T holds complex128 entries.
template <ffi::DataType T>
constexpr bool kComplex = T == ffi::DataType::C128;

// The entries of a value or right-hand-side buffer as doubles (two per complex entry).
template <ffi::DataType T>
const double* Doubles(ffi::Buffer<T>& buffer) {
    return reinterpret_cast<const double*>(buffer.typed_data());
}

// Whether a host slot was last asked to factor exactly `values`, of the kind `complex`.
template <class SlotT>
bool SameValues(const SlotT& slot, const double* values, bool complex) {
    if (!slot.has_values || slot.complex != complex) {
        return false;
    }
    const size_t bytes = slot.values.size() * sizeof(double);
    return std::memcmp(slot.values.data(), values, bytes) == 0;
}

// Refuse a values operand whose length is not the analysed pattern's.
template <class R, ffi::DataType T>
ffi::Error CheckValues(const R&, const typename R::SlotType& slot, ffi::Buffer<T>& values) {
    if (static_cast<int64_t>(values.element_count()) != slot.analysis->nnz_entries) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string(R::kName) + " values operand has " +
                              std::to_string(values.element_count()) +
                              " entries; the analysed pattern has " +
                              std::to_string(slot.analysis->nnz_entries) + ".");
    }
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// The host pattern of an analysis
// --------------------------------------------------------------------------

// The compressed-column pattern of a host analysis, in the index type of its library.
template <class Index>
struct HostPattern {
    int64_t id = 0;
    Index n = 0;
    int64_t nnz_entries = 0;             // caller's coordinate entries
    std::vector<Index> colptr;           // n + 1
    std::vector<Index> rowind;           // compressed-column row indices
    std::vector<int64_t> entry_to_csc;   // coordinate entry -> compressed position
    std::vector<uint8_t> zeroed;         // coordinate entry lies in a Dirichlet row
    std::vector<int64_t> unit_diagonal;  // compressed positions that receive +1
};

// Refuse coordinate arrays of unequal length, an order outside (0, 2**31 - 1] or an entry
// outside the n x n operator. Throws std::invalid_argument, which nanobind raises as
// ValueError.
template <class R>
void CheckCoordinates(int64_t n, int64_t nnz, const int32_t* rows, const int32_t* cols,
                      int64_t cols_length, int64_t zeroed_length) {
    const std::string name(R::kName);
    if (cols_length != nnz || zeroed_length != nnz) {
        throw std::invalid_argument(name +
                                    " analysis needs rows, cols and the Dirichlet mask of equal "
                                    "length.");
    }
    if (n <= 0 || n > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(name + " analysis needs 0 < n <= 2**31 - 1; got n = " +
                                    std::to_string(n) + ".");
    }
    for (int64_t k = 0; k < nnz; ++k) {
        if (rows[k] < 0 || rows[k] >= n || cols[k] < 0 || cols[k] >= n) {
            throw std::invalid_argument(name + " analysis: pattern entry " + std::to_string(k) +
                                        " lies outside the " + std::to_string(n) + " x " +
                                        std::to_string(n) + " operator.");
        }
    }
}

// Fill `a` from checked coordinates: the unique (row, col) pairs in column-major order, the map
// from each coordinate entry to its compressed position, the Dirichlet mask, and the compressed
// position of each Dirichlet row's diagonal, which receives the unit once whatever the
// duplicates. Throws std::invalid_argument for a Dirichlet row outside the operator or without a
// diagonal entry, and for more than 2**31 - 1 distinct entries.
template <class R, class Index>
void BuildPattern(HostPattern<Index>& a, int64_t n, int64_t nnz, const int32_t* r,
                  const int32_t* c, const bool* zeroed, const int32_t* unit_rows,
                  size_t unit_count) {
    const std::string name(R::kName);
    a.n = static_cast<Index>(n);
    a.nnz_entries = nnz;
    a.zeroed.assign(zeroed, zeroed + nnz);
    std::vector<int64_t> order(static_cast<size_t>(nnz));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
        return c[x] != c[y] ? c[x] < c[y] : r[x] < r[y];
    });
    a.entry_to_csc.assign(static_cast<size_t>(nnz), 0);
    a.colptr.assign(static_cast<size_t>(n) + 1, 0);
    a.rowind.clear();
    a.rowind.reserve(static_cast<size_t>(nnz));
    int64_t prev_r = -1, prev_c = -1;
    for (int64_t k : order) {
        if (r[k] != prev_r || c[k] != prev_c) {
            if (a.rowind.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument(
                    name + " analysis: the pattern has more than 2**31 - 1 distinct entries.");
            }
            a.rowind.push_back(static_cast<Index>(r[k]));
            a.colptr[static_cast<size_t>(c[k]) + 1] += 1;
            prev_r = r[k];
            prev_c = c[k];
        }
        a.entry_to_csc[k] = static_cast<int64_t>(a.rowind.size()) - 1;
    }
    for (int64_t j = 0; j < n; ++j) {
        a.colptr[j + 1] += a.colptr[j];
    }
    for (size_t i = 0; i < unit_count; ++i) {
        const Index row = unit_rows[i];
        if (row < 0 || row >= n) {
            throw std::invalid_argument(name + " analysis: Dirichlet row " +
                                        std::to_string(row) + " lies outside the operator.");
        }
        const Index* begin = a.rowind.data() + a.colptr[row];
        const Index* end = a.rowind.data() + a.colptr[row + 1];
        const Index* hit = std::lower_bound(begin, end, row);
        if (hit == end || *hit != row) {
            throw std::invalid_argument(name + " analysis: Dirichlet row " +
                                        std::to_string(row) +
                                        " has no diagonal entry in the pattern.");
        }
        const int64_t position = static_cast<int64_t>(hit - a.rowind.data());
        if (std::find(a.unit_diagonal.begin(), a.unit_diagonal.end(), position) ==
            a.unit_diagonal.end()) {
            a.unit_diagonal.push_back(position);
        }
    }
}

// --------------------------------------------------------------------------
// Python-facing functions
// --------------------------------------------------------------------------

// Drop the registry's reference to an analysis; sites and slots that use it keep it alive.
template <class R>
void AnalysisRelease(int64_t id) {
    nb::gil_scoped_release release;
    R& reg = Instance<R>();
    std::lock_guard<std::mutex> lock(reg.mutex);
    reg.analyses.erase(id);
}

// Create a site on an analysis with an id of its own and return the id.
template <class R>
int64_t SiteCreate(int64_t analysis_id) {
    using SiteT = typename R::SiteType;
    nb::gil_scoped_release release;
    R& reg = Instance<R>();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.analyses.find(analysis_id);
    if (it == reg.analyses.end()) {
        throw std::invalid_argument(std::string(R::kName) + " analysis " +
                                    std::to_string(analysis_id) +
                                    " does not exist in this process; analyse the pattern "
                                    "again.");
    }
    while (reg.sites.count(reg.next_site) != 0) ++reg.next_site;
    auto site = std::make_unique<SiteT>();
    site->id = reg.next_site++;
    site->analysis = it->second;
    const int64_t id = site->id;
    reg.sites.emplace(id, std::move(site));
    return id;
}

// Free every slot of a host context and record the context as released.
template <class R>
void ContextRelease(int64_t context) {
    nb::gil_scoped_release release;
    R& reg = Instance<R>();
    std::lock_guard<std::mutex> lock(reg.mutex);
    ReleaseContextSlots(reg, static_cast<int32_t>(context), [](auto&) {});
}

// Bytes held by the slots of a context: the sum of SlotBytes(slot), which each extension
// defines beside its slot type.
template <class R>
int64_t ResidentBytes(int64_t context) {
    nb::gil_scoped_release release;
    R& reg = Instance<R>();
    std::lock_guard<std::mutex> lock(reg.mutex);
    int64_t total = 0;
    for (const auto& entry : reg.slots) {
        if (entry.second->context == context) {
            total += SlotBytes(*entry.second);
        }
    }
    return total;
}

}  // namespace slot_registry
