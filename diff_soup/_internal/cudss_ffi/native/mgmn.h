// Distributed cuDSS factors with NCCL device and Open MPI host communicators.
// MPI and NCCL are loaded only when an MGMN group is requested.
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <atomic>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace mgmn {
struct UniqueId { char internal[128]; };
inline void Check(int code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}
// A failure that cuDSS reports on a process because another process of the group failed.
struct GroupFailure : std::runtime_error { using std::runtime_error::runtime_error; };
// A cuDSS phase of the group: cuDSS reports a failure local to one process on every process of
// the group as CUDSS_STATUS_EXECUTION_FAILED, so that status names the group failure and points at
// the failing process's own error; any other status is reported as Check reports it.
inline void CheckPhase(cudssStatus_t status, const char* phase, const char* operation) {
    if (status == CUDSS_STATUS_EXECUTION_FAILED)
        throw GroupFailure(std::string("a process of the model group failed in ") + phase +
                           "; its own error names the cause");
    Check(status, operation);
}
template<class F> F Symbol(void* library, const char* name) {
    auto symbol = dlsym(library, name);
    if (!symbol) throw std::runtime_error(std::string("Missing communication symbol: ") + name);
    return reinterpret_cast<F>(symbol);
}
inline void* Open(const char* name) {
    void* library = dlopen(name, RTLD_NOW | RTLD_GLOBAL);
    if (!library) throw std::runtime_error(std::string("Cannot load MGMN dependency ") + name + ": " + dlerror());
    return library;
}
struct Group {
    void* host = nullptr;
    void* device = nullptr;
    int rank = 0;                // this process's rank in the group
    std::vector<int> processes;  // the JAX process index of each rank of the group
    std::string shim;
    std::mutex mutex;
};
inline auto& Groups() {
    static auto* groups = new std::unordered_map<int64_t, std::shared_ptr<Group>>;
    return *groups;
}
inline void*& Mpi() { static void* handle = nullptr; return handle; }
// Whether this library called MPI_Init_thread (it finalizes only an MPI it initialized), and
// whether it has finalized it since.
inline bool& InitializedMpi() { static bool initialized = false; return initialized; }
inline bool& FinalizedMpi() { static bool finalized = false; return finalized; }
inline void*& Nccl() { static void* handle = nullptr; return handle; }
inline void InitializeMpi(const std::string& library) {
    // MPI_Init_thread is collective over the launched world, not a dof group.
    // Call this on every process before assigning independent subgroup work.
    if (!Mpi()) Mpi() = Open(library.c_str());
    int initialized = 0, provided = 0;
    Check(Symbol<int(*)(int*)>(Mpi(), "MPI_Initialized")(&initialized), "MPI_Initialized");
    if (!initialized) {
        if (FinalizedMpi())
            throw std::runtime_error("cuDSS MGMN cannot start: MPI was finalized at interpreter exit.");
        Check(Symbol<int(*)(int*, char***, int, int*)>(Mpi(), "MPI_Init_thread")(nullptr, nullptr, 3, &provided), "MPI_Init_thread");
        InitializedMpi() = true;
    } else Check(Symbol<int(*)(int*)>(Mpi(), "MPI_Query_thread")(&provided), "MPI_Query_thread");
    if (provided < 3) throw std::runtime_error("MGMN requires MPI_THREAD_MULTIPLE.");
}
// The reduction `op` (an Open MPI operation symbol such as "ompi_mpi_op_max") of `value` over the
// processes of `group`, by MPI_Allreduce on its host communicator, so that every process of the
// group takes the same decision from it.
// VERIFY: the Open MPI ABI names ompi_mpi_int64_t, ompi_mpi_op_max and ompi_mpi_op_min of
// MPI_INT64_T, MPI_MAX and MPI_MIN.
inline int64_t GroupReduce(const Group& group, int64_t value, const char* op) {
    int64_t result = value;
    using Allreduce = int (*)(const void*, void*, int, void*, void*, void*);
    Check(Symbol<Allreduce>(Mpi(), "MPI_Allreduce")(&value, &result, 1,
                                                   Symbol<void*>(Mpi(), "ompi_mpi_int64_t"),
                                                   Symbol<void*>(Mpi(), op), group.host),
          "MPI_Allreduce");
    return result;
}
inline int64_t GroupMax(const Group& group, int64_t value) { return GroupReduce(group, value, "ompi_mpi_op_max"); }
// `count` bytes at `buffer` of the group's rank `root` on every process of the group (MPI_Bcast on
// its host communicator).
inline void GroupBroadcast(const Group& group, void* buffer, int64_t count, int root) {
    if (count < 1) return;
    Check(Symbol<int(*)(void*, int, void*, int, void*)>(Mpi(), "MPI_Bcast")(
              buffer, static_cast<int>(count), Symbol<void*>(Mpi(), "ompi_mpi_byte"), root, group.host),
          "MPI_Bcast");
}
// Every process of `group` agrees whether `step` succeeded on all of them, before any of them
// enters the group's next collective call: a process that failed and returned would otherwise
// leave the others blocked in that call. `error` is this process's error (empty on success);
// `invalid` marks an invalid argument and `secondary` a failure cuDSS reports because another
// process failed. When some process failed, every process throws the same error: the error of
// the first rank that failed for a cause of its own (else of the first rank that failed), with
// that rank and its JAX process named. A failure inside a collective call of cuDSS that leaves
// the other processes waiting in it cannot be reported this way.
inline void Agree(const Group& group, const char* step, const std::string& error, bool invalid, bool secondary) {
    const int64_t size = static_cast<int64_t>(group.processes.size());
    const int64_t key = error.empty() ? 2 * size : (secondary ? size : 0) + group.rank;
    const int64_t first = GroupReduce(group, key, "ompi_mpi_op_min");
    if (first == 2 * size) return;
    const int root = static_cast<int>(first % size);
    int64_t header[2] = {invalid ? 1 : 0, static_cast<int64_t>(error.size())};
    GroupBroadcast(group, header, sizeof(header), root);
    std::string text = group.rank == root ? error : std::string(static_cast<size_t>(header[1]), '\0');
    GroupBroadcast(group, text.data(), header[1], root);
    const std::string message = "cuDSS MGMN: rank " + std::to_string(root) + " of the model group (JAX process " +
                                std::to_string(group.processes[root]) + ") failed in " + step + ": " + text;
    if (header[0]) throw std::invalid_argument(message);
    throw std::runtime_error(message);
}
// Runs `body` on this process, then agrees with the group on its outcome (see Agree).
template<class F> void Step(const Group& group, const char* step, F&& body) {
    std::string error;
    bool invalid = false, secondary = false;
    try { body(); }
    catch (const GroupFailure& failure) { error = failure.what(); secondary = true; }
    catch (const std::invalid_argument& failure) { error = failure.what(); invalid = true; }
    catch (const std::exception& failure) { error = failure.what(); }
    Agree(group, step, error, invalid, secondary);
}
inline nb::bytes NewId(const std::string& library) {
    if (!Nccl()) Nccl() = Open(library.c_str());
    UniqueId id;
    Check(Symbol<int(*)(UniqueId*)>(Nccl(), "ncclGetUniqueId")(&id), "ncclGetUniqueId");
    return nb::bytes(id.internal, sizeof(id));
}
// The MPI host communicator of the group `id` over the world ranks `ranks` (in that order), made
// at the group's first call (MPI_Comm_create_group, collective over those processes only) and kept
// for its broadcast and its cuDSS factors. MPI is initialized.
inline void* HostCommunicator(int64_t id, const std::vector<int>& ranks, int process) {
    static auto* communicators = new std::unordered_map<int64_t, void*>;
    auto found = communicators->find(id);
    if (found != communicators->end()) return found->second;
    if (ranks.empty()) throw std::runtime_error("MGMN needs nonempty ranks.");
    int rank = -1, size = 0;
    void* world = Symbol<void*>(Mpi(), "ompi_mpi_comm_world");
    Check(Symbol<int(*)(void*, int*)>(Mpi(), "MPI_Comm_rank")(world, &rank), "MPI_Comm_rank");
    Check(Symbol<int(*)(void*, int*)>(Mpi(), "MPI_Comm_size")(world, &size), "MPI_Comm_size");
    if (*std::min_element(ranks.begin(), ranks.end()) < 0 || *std::max_element(ranks.begin(), ranks.end()) >= size)
        throw std::runtime_error("MPI world does not contain the JAX dof group; use the Slurm PMI2 launcher.");
    if (rank != process) throw std::runtime_error("MPI and JAX ranks differ; launch both through the same Slurm PMI2 step.");
    if (std::find(ranks.begin(), ranks.end(), process) == ranks.end()) throw std::runtime_error("This process does not own the MGMN group.");
    void *world_group = nullptr, *subgroup = nullptr, *host = nullptr;
    Check(Symbol<int(*)(void*, void**)>(Mpi(), "MPI_Comm_group")(world, &world_group), "MPI_Comm_group");
    Check(Symbol<int(*)(void*, int, const int*, void**)>(Mpi(), "MPI_Group_incl")(world_group, ranks.size(), ranks.data(), &subgroup), "MPI_Group_incl");
    Check(Symbol<int(*)(void*, void*, int, void**)>(Mpi(), "MPI_Comm_create_group")(world, subgroup, id % 32767, &host), "MPI_Comm_create_group");
    Check(Symbol<int(*)(void**)>(Mpi(), "MPI_Group_free")(&subgroup), "MPI_Group_free");
    Check(Symbol<int(*)(void**)>(Mpi(), "MPI_Group_free")(&world_group), "MPI_Group_free");
    communicators->emplace(id, host);
    return host;
}
// The `bytes` of the group's first process (rank 0 of its host communicator) on every process of
// the group, by MPI_Bcast on the host: the processes agree without a JAX collective, so the call
// may run while a JAX program is traced. Every process of `ranks` calls it in the same order with
// bytes of one length.
inline nb::bytes Broadcast(int64_t id, const std::vector<int>& ranks, int process,
                           const nb::bytes& bytes, const std::string& mpi_library) {
    std::string buffer(bytes.c_str(), bytes.size());
    {
        nb::gil_scoped_release release;
        InitializeMpi(mpi_library);
        void* host = HostCommunicator(id, ranks, process);
        Check(Symbol<int(*)(void*, int, void*, int, void*)>(Mpi(), "MPI_Bcast")(
                  buffer.data(), static_cast<int>(buffer.size()), Symbol<void*>(Mpi(), "ompi_mpi_byte"), 0, host),
              "MPI_Bcast");
    }
    return nb::bytes(buffer.data(), buffer.size());
}
inline void Initialize(int64_t id, const std::vector<int>& ranks, int process,
                       const nb::bytes& bytes, const std::string& nccl_library,
                       const std::string& mpi_library, const std::string& shim) {
    if (Groups().count(id)) return;
    if (bytes.size() != 128 || ranks.empty()) throw std::runtime_error("MGMN needs a 128-byte NCCL id and nonempty ranks.");
    UniqueId unique;
    std::memcpy(&unique, bytes.c_str(), sizeof(unique));
    nb::gil_scoped_release release;
    InitializeMpi(mpi_library);
    auto group = std::make_shared<Group>();
    group->host = HostCommunicator(id, ranks, process);
    group->processes = ranks;
    group->rank = static_cast<int>(std::find(ranks.begin(), ranks.end(), process) - ranks.begin());
    // ncclCommInitRank waits for every rank of the group: every process loads NCCL first.
    Step(*group, "loading NCCL", [&] { if (!Nccl()) Nccl() = Open(nccl_library.c_str()); });
    Step(*group, "the NCCL communicator setup", [&] {
        Check(Symbol<int(*)(void**, int, UniqueId, int)>(Nccl(), "ncclCommInitRank")(&group->device, ranks.size(), unique, group->rank), "ncclCommInitRank");
    });
    group->shim = shim;
    Groups().emplace(id, group);
}
struct Factor {
    std::shared_ptr<Group> group;
    cudssHandle_t handle = nullptr;
    cudssData_t data = nullptr;
    cudssConfig_t config = nullptr;
    cudssMatrix_t a = nullptr, b = nullptr, x = nullptr;
    void *offsets = nullptr, *columns = nullptr, *values = nullptr, *rhs = nullptr, *solution = nullptr;
    int64_t n = 0, nnz = 0, nrhs = 0, stamp[2] = {0, 0};
    // Distributed input: this process hands cuDSS the rows [first_row, last_row] of the matrix
    // and of the right-hand side (local_nnz entries of the matrix, from entry value_offset of the
    // CSR values); with central input the first process hands it every row.
    bool distributed = false;
    int64_t first_row = 0, last_row = -1, local_nnz = 0, value_offset = 0;
    // The bytes of the matrix and of the last solve's right-hand side handed to cuDSS here.
    int64_t matrix_bytes = 0, rhs_bytes = 0;
    int device_id = 0;
    int64_t instance_id = 0;
    // Per-call factors: the factor's number in the order the group made them (the same on every
    // process), and whether the program that held it has dropped it (see GetRaw).
    int64_t serial = 0;
    std::atomic<bool> retired{false};
    bool analyzed = false, factored = false, stamped = false;
    cudssSolveMode_t solve_mode = CUDSS_SOLVE_MODE_NORMAL;  // the mode `config` holds
    int64_t analysis_calls = 0, factor_calls = 0, solve_calls = 0;
    double analysis_ms = 0, factor_ms = 0, solve_ms = 0;
    int64_t memory[16] = {};
    ~Factor() { Release(); }
    // Destroys the cuDSS objects and frees the buffers of the factor, which stays an empty record.
    void Release() {
        cudaSetDevice(device_id);
        if (a) cudssMatrixDestroy(a);
        if (b) cudssMatrixDestroy(b);
        if (x) cudssMatrixDestroy(x);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        for (void* buffer : {offsets, columns, values, rhs, solution}) if (buffer) cudaFree(buffer);
        a = b = x = nullptr;
        data = nullptr;
        config = nullptr;
        handle = nullptr;
        offsets = columns = values = rhs = solution = nullptr;
        nrhs = 0;
        analyzed = factored = stamped = false;
        solve_mode = CUDSS_SOLVE_MODE_NORMAL;
    }
};
// The solve mode of the group's configuration (CUDSS_CONFIG_SOLVE_MODE): transpose for a solve
// with the transpose of the factored matrix, normal for every other phase. Every process of the
// group sets the same mode, since every one runs the same program.
inline void SetSolveMode(Factor& f, cudssSolveMode_t mode) {
    if (f.solve_mode == mode) return;
    Check(cudssConfigSet(f.config, CUDSS_CONFIG_SOLVE_MODE, &mode, sizeof(mode)), "set MGMN solve mode");
    f.solve_mode = mode;
}
// The per-call factors, owned here in the order the group made them (the same on every process:
// a program's first call); a program's state holds a lease on its factor (see GetRaw).
template<ffi::DataType T> auto& RawFactors() {
    static auto* factors = new std::vector<std::shared_ptr<Factor>>;
    return *factors;
}
// Destroys the retired per-call factors of `group` that every process of the group has retired,
// in the order the group made them. Destroying a factor's cuDSS objects is collective over the
// group (the communication layer destroys NCCL communicators, each with a barrier among the
// processes), so the processes first agree on the factors to destroy: an allgather of each
// process's retired factors over the host communicator. Called at a collective point of the group
// (the creation of a new factor, reached by every process in the same program order); a factor
// retired on this process only waits for a later point or for the finalization at exit.
template<ffi::DataType T> void ReleaseRetired(Group& group) {
    std::lock_guard<std::mutex> lock(group.mutex);
    auto& factors = RawFactors<T>();
    std::vector<int64_t> mine;
    for (const auto& f : factors) if (f->group.get() == &group && f->retired.load()) mine.push_back(f->serial);
    const int64_t size = static_cast<int64_t>(group.processes.size());
    const int64_t count = GroupMax(group, static_cast<int64_t>(mine.size()));
    if (count == 0) return;
    std::vector<int64_t> sent(count, -1), all(count * size, -1);
    std::copy(mine.begin(), mine.end(), sent.begin());
    using Allgather = int (*)(const void*, int, void*, void*, int, void*, void*);
    void* int64 = Symbol<void*>(Mpi(), "ompi_mpi_int64_t");
    Check(Symbol<Allgather>(Mpi(), "MPI_Allgather")(sent.data(), static_cast<int>(count), int64, all.data(),
                                                     static_cast<int>(count), int64, group.host), "MPI_Allgather");
    std::vector<int64_t> common;
    for (int64_t serial : mine) {
        bool everywhere = true;
        for (int64_t r = 0; r < size && everywhere; ++r) {
            const auto first = all.begin() + r * count, last = first + count;
            everywhere = std::find(first, last, serial) != last;
        }
        if (everywhere) common.push_back(serial);
    }
    std::sort(common.begin(), common.end());
    for (int64_t serial : common) {
        auto it = std::find_if(factors.begin(), factors.end(), [&](const auto& f) { return f->serial == serial; });
        if (it == factors.end()) continue;
        (*it)->Release();
        factors.erase(it);
    }
}
// The per-call factor of `instance` held by a program's `state`, made at the program's first
// call. The state's pointer does not own the factor: when the program is dropped (on each process
// in its own order, as its caches are cleared), the state only retires the factor, and the group
// destroys the factors every process has retired at its next collective point (ReleaseRetired)
// or at the finalization at exit.
template<ffi::DataType T> Factor& GetRaw(int64_t instance, int64_t group_id,
                                       std::shared_ptr<Factor>& state) {
    if (FinalizedMpi()) throw std::runtime_error("cuDSS MGMN cannot solve: MPI was finalized at interpreter exit.");
    if (!state) {
        if (!Groups().count(group_id)) throw std::runtime_error("MGMN communicator is not initialized.");
        static int64_t serials = 0;
        auto factor = std::make_shared<Factor>();
        factor->group = Groups().at(group_id);
        factor->instance_id = instance;
        factor->serial = ++serials;
        Check(cudaGetDevice(&factor->device_id), "cudaGetDevice");
        state = std::shared_ptr<Factor>(factor.get(), [](Factor* f) { f->retired.store(true); });
        RawFactors<T>().push_back(factor);
        ReleaseRetired<T>(*factor->group);
    }
    return *state;
}
// The rows of the right-hand side this process hands cuDSS: its own row block with distributed
// input, every row on the group's first process with central input, none elsewhere.
inline int64_t RhsRows(const Factor& f) {
    if (f.distributed) return f.last_row - f.first_row + 1;
    return f.group->rank == 0 ? f.n : 0;
}
// The right-hand-side matrix of `f` with `columns` columns over a new, zeroed `f.rhs` of this
// process's rows (RhsRows): the global shape on every process, and with distributed input the
// row distribution of this process's block, as the matrix has (measured with one column on
// cuDSS 1.0.0, two processes, the NCCL layer).
template<class Scalar> void CreateRhs(Factor& f, int64_t columns, cudaDataType_t type) {
    const int64_t entries = RhsRows(f) * columns;
    if (entries) {
        Check(cudaMalloc(&f.rhs, entries * sizeof(Scalar)), "allocate MGMN RHS");
        Check(cudaMemset(f.rhs, 0, entries * sizeof(Scalar)), "clear MGMN RHS");
    }
    Check(cudssMatrixCreateDn(&f.b, f.n, columns, f.n, f.rhs, type, CUDSS_LAYOUT_COL_MAJOR), "create MGMN RHS");
    if (f.distributed) Check(cudssMatrixSetDistributionRow1d(f.b, f.first_row, f.last_row), "distribute MGMN RHS rows");
}
// The solution matrix of `f` over `f.solution` with `columns` columns, replicated: its row
// distribution is the whole range [0, n - 1] on every process of the group, so cuDSS writes the
// full solution on every process, with central and with distributed input. Measured equal to a
// solution on the first process alone with one right-hand side and both solve modes (cuDSS
// 1.0.0, two processes). VERIFY: several right-hand sides.
inline void CreateSolution(Factor& f, int64_t columns, cudaDataType_t type) {
    Check(cudssMatrixCreateDn(&f.x, f.n, columns, f.n, f.solution, type, CUDSS_LAYOUT_COL_MAJOR), "create MGMN solution");
    Check(cudssMatrixSetDistributionRow1d(f.x, 0, f.n - 1), "replicate MGMN solution");
}
// Every collective call of a factor (cuDSS's analysis, factorization, refactorization and solve,
// and the hybrid-memory query) is preceded by a Step that runs the process's own preparation and
// agrees on its outcome, and is itself run as a Step, so a process that fails stops the others
// before their next collective call and every process raises the same error.
template<ffi::DataType T> void Refactor(Factor& f, cudaStream_t stream, const void* values,
        ffi::Span<const int32_t> offsets, ffi::Span<const int32_t> columns,
        int mtype_id, int mview_id, int ir, int reorder, int matching,
        int hybrid, int64_t limit, int superpanels, bool distributed) {
    using Scalar = typename get_native_data_type<T>::type;
    const Group& group = *f.group;
    const bool root = group.rank == 0;
    Step(group, "the factorization input", [&] {
        if (mtype_id < 0 || mtype_id >= 5 || mview_id < 0 || mview_id >= 3)
            throw std::invalid_argument("cuDSS MGMN takes a matrix type id in [0, 4] and a view id in [0, 2]; got " +
                                        std::to_string(mtype_id) + " and " + std::to_string(mview_id) + ".");
        Check(cudaStreamSynchronize(stream), "MGMN input synchronization");
        if (!f.handle) {
            f.n = offsets.size() - 1; f.nnz = columns.size();
            if (f.n < 1 || f.nnz < 1)
                throw std::invalid_argument("cuDSS MGMN needs a CSR pattern with at least one row and one entry; got " +
                                            std::to_string(f.n) + " rows and " + std::to_string(f.nnz) + " entries.");
            Check(cudssCreate(&f.handle), "cudssCreate MGMN");
            Check(cudssSetStream(f.handle, stream), "cudssSetStream MGMN");
            Check(cudssSetCommLayer(f.handle, group.shim.c_str()), "cudssSetCommLayer");
            Check(cudssDataCreate(f.handle, &f.data), "cudssDataCreate MGMN");
            Check(cudssDataSet(f.handle, f.data, CUDSS_DATA_COMM_DEVICE, &f.group->device, sizeof(void*)), "set device communicator");
            Check(cudssDataSet(f.handle, f.data, CUDSS_DATA_COMM_HOST, &f.group->host, sizeof(void*)), "set host communicator");
            Check(cudssConfigCreate(&f.config), "cudssConfigCreate MGMN");
            Check(cudssConfigSet(f.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)), "set MGMN refinement");
            if (reorder >= 0) Check(cudssConfigSet(f.config, CUDSS_CONFIG_REORDERING_ALG, &reorder, sizeof(reorder)), "set MGMN ordering");
            if (matching >= 0) Check(cudssConfigSet(f.config, CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof(matching)), "set MGMN matching");
            if (superpanels >= 0) Check(cudssConfigSet(f.config, CUDSS_CONFIG_USE_SUPERPANELS, &superpanels, sizeof(superpanels)), "set MGMN superpanels");
            // Hybrid memory before the analysis; its device limit after it (below).
            if (hybrid) Check(cudssConfigSet(f.config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hybrid, sizeof(hybrid)), "set MGMN hybrid memory");
            f.distributed = distributed;
            if (distributed) {
                // This process's contiguous block of rows, sliced from the whole pattern and
                // values every process holds: rank r of a group of p processes takes the rows
                // [n r / p, n (r + 1) / p - 1].
                const int64_t size = static_cast<int64_t>(group.processes.size());
                if (f.n < size)
                    throw std::invalid_argument("cuDSS MGMN with distributed input gives every process of the model group at "
                                                "least one row; got " + std::to_string(f.n) + " rows for " +
                                                std::to_string(size) + " processes. Use central input.");
                f.first_row = f.n * group.rank / size;
                f.last_row = f.n * (group.rank + 1) / size - 1;
                f.value_offset = offsets[f.first_row];
                f.local_nnz = offsets[f.last_row + 1] - f.value_offset;
                std::vector<int32_t> local(offsets.begin() + f.first_row, offsets.begin() + f.last_row + 2);
                for (auto& offset : local) offset -= static_cast<int32_t>(f.value_offset);
                Check(cudaMalloc(&f.offsets, local.size() * sizeof(int32_t)), "allocate MGMN offsets");
                Check(cudaMalloc(&f.columns, std::max<int64_t>(f.local_nnz, 1) * sizeof(int32_t)), "allocate MGMN columns");
                Check(cudaMalloc(&f.values, std::max<int64_t>(f.local_nnz, 1) * sizeof(Scalar)), "allocate MGMN values");
                Check(cudaMemcpy(f.offsets, local.data(), local.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy MGMN offsets");
                Check(cudaMemcpy(f.columns, columns.begin() + f.value_offset, f.local_nnz * sizeof(int32_t), cudaMemcpyHostToDevice), "copy MGMN columns");
                f.matrix_bytes = static_cast<int64_t>(local.size() * sizeof(int32_t)) + f.local_nnz * static_cast<int64_t>(sizeof(int32_t) + sizeof(Scalar));
            } else if (root) {
                f.first_row = 0;
                f.last_row = f.n - 1;
                f.local_nnz = f.nnz;
                Check(cudaMalloc(&f.offsets, offsets.size() * sizeof(int32_t)), "allocate MGMN offsets");
                Check(cudaMalloc(&f.columns, columns.size() * sizeof(int32_t)), "allocate MGMN columns");
                Check(cudaMalloc(&f.values, f.nnz * sizeof(Scalar)), "allocate MGMN values");
                Check(cudaMemcpy(f.offsets, offsets.begin(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy MGMN offsets");
                Check(cudaMemcpy(f.columns, columns.begin(), columns.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy MGMN columns");
                f.matrix_bytes = static_cast<int64_t>(offsets.size() * sizeof(int32_t)) + f.nnz * static_cast<int64_t>(sizeof(int32_t) + sizeof(Scalar));
            }
            const cudssMatrixType_t types[] = {CUDSS_MTYPE_GENERAL, CUDSS_MTYPE_SYMMETRIC, CUDSS_MTYPE_HERMITIAN, CUDSS_MTYPE_SPD, CUDSS_MTYPE_HPD};
            const cudssMatrixViewType_t views[] = {CUDSS_MVIEW_FULL, CUDSS_MVIEW_UPPER, CUDSS_MVIEW_LOWER};
            // The global shape and entry count on every process; with distributed input the
            // arrays hold this process's rows, with offsets from its first row and global
            // column indices (measured on cuDSS 1.0.0, two processes, the NCCL layer).
            Check(cudssMatrixCreateCsr(&f.a, f.n, f.n, f.nnz, f.offsets, nullptr, f.columns, f.values,
                CUDA_R_32I, CUDA_R_32I, get_cuda_data_type<T>(), types[mtype_id], views[mview_id], CUDSS_BASE_ZERO), "create MGMN CSR");
            if (distributed) Check(cudssMatrixSetDistributionRow1d(f.a, f.first_row, f.last_row), "distribute MGMN matrix rows");
        }
        Check(cudssSetStream(f.handle, stream), "set MGMN factor stream");
        if (f.distributed) {
            if (f.local_nnz)
                Check(cudaMemcpyAsync(f.values, static_cast<const Scalar*>(values) + f.value_offset, f.local_nnz * sizeof(Scalar),
                                      cudaMemcpyDeviceToDevice, stream), "copy MGMN values");
        } else if (root) Check(cudaMemcpyAsync(f.values, values, f.nnz * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream), "copy MGMN values");
        Check(cudaStreamSynchronize(stream), "MGMN value synchronization");
        if (!f.analyzed && !f.x) {
            CreateRhs<Scalar>(f, 1, get_cuda_data_type<T>());
            Check(cudaMalloc(&f.solution, f.n * sizeof(Scalar)), "allocate MGMN analysis solution");
            Check(cudaMemset(f.solution, 0, f.n * sizeof(Scalar)), "clear MGMN analysis solution");
            CreateSolution(f, 1, get_cuda_data_type<T>());
            f.nrhs = 1;
        }
        SetSolveMode(f, CUDSS_SOLVE_MODE_NORMAL);
    });
    if (!f.analyzed) {
        Step(group, "the analysis", [&] {
            auto start = Clock::now();
            CheckPhase(cudssExecute(f.handle, CUDSS_PHASE_ANALYSIS, f.config, f.data, f.a, f.x, f.b), "the analysis", "MGMN analysis");
            f.analysis_ms += ElapsedMs(start); ++f.analysis_calls;
            size_t written = 0;
            Check(cudssDataGet(f.handle, f.data, CUDSS_DATA_MEMORY_ESTIMATES, f.memory, sizeof(f.memory), &written), "MGMN memory estimates");
        });
        if (hybrid && limit > 0) {
            // The device floor of the analysed factorization (a collective query, made on every
            // rank), the largest over the group so every rank decides alike; a limit below it is
            // refused before the factorization, which would reject it.
            int64_t minimum = -1;
            Step(group, "the hybrid memory query", [&] {
                size_t written = 0;
                Check(cudssDataGet(f.handle, f.data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, &minimum, sizeof(minimum), &written), "MGMN hybrid memory minimum");
            });
            minimum = GroupMax(group, minimum);
            const std::string refusal = HybridLimitRefusal(minimum, limit);
            if (!refusal.empty()) throw std::invalid_argument(refusal);
            Step(group, "the hybrid memory limit", [&] {
                int64_t bytes = limit * 1024 * 1024;
                Check(cudssConfigSet(f.config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT, &bytes, sizeof(bytes)), "set MGMN memory limit");
            });
        }
        f.analyzed = true;
    }
    const char* phase = f.factored ? "the refactorization" : "the factorization";
    Step(group, phase, [&] {
        auto start = Clock::now();
        CheckPhase(cudssExecute(f.handle, f.factored ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION,
            f.config, f.data, f.a, f.x, f.b), phase, "MGMN factorization");
        f.factor_ms += ElapsedMs(start); ++f.factor_calls;
        Check(cudaStreamSynchronize(stream), "MGMN factor synchronization");
    });
    f.factored = true; f.stamped = false;
}
// With distributed input a block of right-hand sides is solved one column at a time, each with
// this process's rows of that column. VERIFY: the layout of a distributed block of several
// right-hand sides (a leading dimension of the local rows or of n), which would solve the block
// in one call.
template<ffi::DataType T> void Solve(Factor& f, cudaStream_t stream, const void* rhs, void* output, int64_t elements,
                                    cudssSolveMode_t mode) {
    using Scalar = typename get_native_data_type<T>::type;
    const Group& group = *f.group;
    const int64_t columns = f.n > 0 ? elements / f.n : 0;
    const int64_t width = f.distributed ? 1 : columns;  // the columns of one cuDSS solve
    // This process's rows of the `width` columns from column `first` into f.rhs, and, except in
    // float64, a cleared solution.
    auto stage = [&](int64_t first) {
        const int64_t rows = RhsRows(f);
        if (rows == f.n) {
            Check(cudaMemcpyAsync(f.rhs, static_cast<const Scalar*>(rhs) + first * f.n, width * f.n * sizeof(Scalar),
                                  cudaMemcpyDeviceToDevice, stream), "copy MGMN RHS");
        } else if (rows) {
            Check(cudaMemcpyAsync(f.rhs, static_cast<const Scalar*>(rhs) + first * f.n + f.first_row, rows * sizeof(Scalar),
                                  cudaMemcpyDeviceToDevice, stream), "copy MGMN RHS");
        }
        f.rhs_bytes = rows * columns * static_cast<int64_t>(sizeof(Scalar));
        if constexpr (ZeroFillsSolveOutput<T>())  // float64 solves skip the clear (see the helper)
            Check(cudaMemsetAsync(f.solution, 0, width * f.n * sizeof(Scalar), stream), "clear MGMN solution");
        Check(cudaStreamSynchronize(stream), "MGMN solve input synchronization");
    };
    Step(group, "the solve input", [&] {
        if (!f.factored || elements < 1 || elements % f.n)
            throw std::invalid_argument("cuDSS MGMN solves with a factored matrix and whole right-hand-side columns of " +
                                        std::to_string(f.n) + " entries; got " + std::to_string(elements) + " entries.");
        Check(cudaStreamSynchronize(stream), "MGMN RHS synchronization");
        if (f.nrhs != width) {
            if (f.b) Check(cudssMatrixDestroy(f.b), "destroy MGMN RHS");
            if (f.x) Check(cudssMatrixDestroy(f.x), "destroy MGMN solution");
            if (f.rhs) Check(cudaFree(f.rhs), "free MGMN RHS");
            if (f.solution) Check(cudaFree(f.solution), "free MGMN solution");
            f.b = f.x = nullptr;
            f.rhs = f.solution = nullptr;
            f.nrhs = 0;
            CreateRhs<Scalar>(f, width, get_cuda_data_type<T>());
            Check(cudaMalloc(&f.solution, width * f.n * sizeof(Scalar)), "allocate MGMN solution");
            CreateSolution(f, width, get_cuda_data_type<T>());
            f.nrhs = width;
        }
        Check(cudssSetStream(f.handle, stream), "set MGMN solve stream");
        stage(0);
        SetSolveMode(f, mode);
    });
    for (int64_t first = 0; first < columns; first += width) {
        if (first) Step(group, "the solve input", [&] { stage(first); });
        Step(group, "the solve", [&] {
            auto start = Clock::now();
            CheckPhase(cudssExecute(f.handle, CUDSS_PHASE_SOLVE, f.config, f.data, f.a, f.x, f.b), "the solve", "MGMN solve");
            f.solve_ms += ElapsedMs(start); ++f.solve_calls;
            Check(cudaMemcpyAsync(static_cast<Scalar*>(output) + first * f.n, f.solution, width * f.n * sizeof(Scalar),
                                  cudaMemcpyDeviceToDevice, stream), "copy MGMN solution");
            Check(cudaStreamSynchronize(stream), "MGMN output synchronization");
        });
    }
}
template<ffi::DataType T> void ReleaseFactors() {
    for (const auto& f : RawFactors<T>()) {
        std::lock_guard<std::mutex> lock(f->group->mutex);
        f->Release();
    }
}
// Finalizes MPI when this library initialized it and it is not finalized yet; returns whether it
// did. The cuDSS objects of every MGMN factor are destroyed first, in the order the factors were
// made (the same on every process), so no factor calls into MPI after the finalization when JAX
// later destroys the programs that hold it; a factor used afterwards raises an error.
inline bool FinalizeMpi() {
    if (!Mpi() || !InitializedMpi() || FinalizedMpi()) return false;
    int finalized = 0;
    Check(Symbol<int(*)(int*)>(Mpi(), "MPI_Finalized")(&finalized), "MPI_Finalized");
    if (finalized) return false;
    FinalizedMpi() = true;
    ReleaseFactors<ffi::F64>();
    ReleaseFactors<ffi::F32>();
    Check(Symbol<int(*)()>(Mpi(), "MPI_Finalize")(), "MPI_Finalize");
    return true;
}
inline void Export(nb::module_& m) {
    m.def("mgmn_initialize_mpi", [](const std::string& library) {
        nb::gil_scoped_release release;
        InitializeMpi(library);
    });
    m.def("mgmn_finalize_mpi", []() {
        nb::gil_scoped_release release;
        return FinalizeMpi();
    });
    m.def("mgmn_unique_id", &NewId);
    m.def("mgmn_initialize", &Initialize);
    m.def("mgmn_broadcast", &Broadcast);
    m.def("mgmn_stats", []() {
        nb::list result;
        auto append = [&](const Factor& f) {
            nb::dict record;
            record["instance_id"] = f.instance_id;
            record["rank"] = f.group->rank;
            record["input"] = f.distributed ? "distributed" : "central";
            record["retired"] = f.retired.load();
            record["first_row"] = f.first_row;
            record["last_row"] = f.last_row;
            // The bytes of the CSR matrix (offsets, columns, values) and of the last solve's
            // right-hand side that this process hands cuDSS.
            record["input_bytes"] = f.matrix_bytes + f.rhs_bytes;
            record["analysis_calls"] = f.analysis_calls;
            record["factor_calls"] = f.factor_calls;
            record["solve_calls"] = f.solve_calls;
            record["analysis_ms"] = f.analysis_ms;
            record["factor_ms"] = f.factor_ms;
            record["solve_ms"] = f.solve_ms;
            // cuDSS's own device memory estimates after the analysis (CUDSS_DATA_MEMORY_ESTIMATES,
            // entries 0 and 1) on this process, not a measurement of the device's free memory.
            record["device_permanent_bytes"] = f.memory[0];
            record["device_peak_bytes"] = f.memory[1];
            result.append(record);
        };
        for (const auto& factor : RawFactors<ffi::F64>()) append(*factor);
        return result;
    });
}
}  // namespace mgmn
