// cuDSS factors held in native slots that a SolveContext owns, exposed to JAX through XLA FFI
// handlers for the phased factor, solve, refactor and invalidate primitives of
// diff_soup.linear.
//
// Included by single_solve.cpp after the cuDSS helpers it calls (CudssCallFailed,
// ZeroSolveOutput, ApplyDiagnosticConfig, QueryFactorFacts, QueryMemoryEstimates,
// HybridLimitRefusal, SetHybridDeviceLimit); BindContract(m) adds the Python-facing functions to
// the module.
//
// Objects:
//   * an analysis per coordinate pattern and options: the CSR pattern of the operator (offsets
//     and columns, host copies for the life of the record), the map that assembles the caller's
//     coordinate values into that CSR order (assembly_kernels.h), the cuDSS options, the
//     fill-reducing permutation of the first cuDSS analysis run on it, and the device copies of
//     the pattern and map, uploaded on first use on each device and
//     shared by every slot on that device. The device copies are reference counted per
//     context: the first slot of a context takes a reference, releasing the context drops it,
//     and at zero the device copies on every device are freed and uploaded again on next use
//     under the same analysis id;
//   * a slot per factor: one cuDSS handle and its single side: one cuDSS configuration, data
//     object, CSR matrix over the shared device pattern of A_bc, device copy of the CSR values
//     the factor was built from (iterative refinement reads the matrix at solve time, after the
//     caller's buffer has expired), the dense right-hand side and solution matrices re-pointed
//     at every solve, a generation and a status. The side factors A_bc whatever the slot's
//     orientation, and serves both orientations from that one factor: a solve with A_bc^T (a
//     transposed solve of an "A" factor, or a solve of an "AT" factor) runs cuDSS's solve phase
//     in transpose mode (CUDSS_CONFIG_SOLVE_MODE = CUDSS_SOLVE_MODE_TRANSPOSE, counted in
//     transposed_solves). A slot declared symmetric or symmetric positive definite solves
//     both orientations in normal mode, since its matrix is its transpose;
//   * a batch slot per factor built under a native vmap, whose values carry a leading lane
//     axis: one slot holds the factors of every lane in tiles of consecutive lanes. In the
//     default stacked layout a tile is one cuDSS system of lanes * n rows whose matrix holds
//     the lanes block-diagonally (the lane-major value copy is its value array, and lane-major
//     right-hand sides of one column per lane are its right-hand side); in the uniform layout
//     (batch_layout) a tile is a cuDSS uniform batch (one pattern, CUDSS_CONFIG_UBATCH_SIZE
//     lanes of values one after the other). Each tile holds as many consecutive lanes as keep
//     its row and value counts within int32 and its factor entries within kBatchFactorEntries.
//     A batch slot of L lanes reserves L consecutive
//     slot ids, one per lane, so its handle leaves are vectors of L entries: the context id, each
//     lane's slot id and each lane's generation. The batch side keeps one device copy of every
//     lane's CSR values (lane i at offset i * nse) and every lane's status. Like a single side it
//     factors A_bc whatever the slot's orientation and solves with A_bc^T in cuDSS's transpose
//     solve mode;
//   * a site per lowered factor or refactor call: the slots it writes are keyed at the site by
//     the context id the call carries at run time and the lane token (factor) or the held slot
//     (refactor). A refactor site also records, per context id and held slot, that its call
//     returned the held handle unchanged (a false predicate). A call names its site by a key,
//     a deterministic 64-bit hash of the lowered call that the lowering computes, and the site
//     is created on the first call that carries an unknown key, so two lowerings of one program
//     share their sites and their program text. Sites belong to no context and live for the
//     process;
//   * shared storage: a slot named by (context id, device, analysis, precision, lane count,
//     storage key) instead of by a site, for the factors that empty_factor(..., storage=...)
//     starts. Every call site and program that refactors such a factor on one device writes
//     that one slot, and the factor trusts it (see "Shared storage" below); each device that
//     runs the program keeps a slot of its own.
//
// Values: every call takes the operator values as the caller passes them, in the coordinate
// order of the topology (one entry per coordinate, duplicates included, float32 or float64),
// and assembles them on the device into the CSR values of the side that uses them (duplicates
// added, every entry of a Dirichlet row multiplied by zero and one added at the row's
// diagonal, cast to the slot's dtype) with one kernel per run of lanes, straight into the
// side's value copy: a factor and a refactor for the lanes they factor, a solve only for a lane
// it must factor before solving. The lowered programs carry no CSR assembly and no index
// constant of the pattern.
//
// A factor handle is three int32 scalars (context id, slot id, generation), or three vectors
// for a batch slot. Every factorization request (factor, refactor) gives its slot a new
// generation and writes it into the handle (in a batch slot, into the lanes it factors anew; a
// lane that did not ask keeps its own). A solve whose handle generation differs from the
// generation of the side it uses factors that side (that lane, in a batch) again from the
// values the solve carries, so two live factors that share a slot are never confused. Before
// such a rebuild of a side that holds a factor, the values are assembled into the side's copy by
// a kernel that
// compares every entry with the one it replaces and marks the lanes that changed, read back
// with one copy: a lane whose values are unchanged already holds the handle's factor and takes
// its generation without a factorization (counted in rebuilds_avoided).
//
// Shared storage: a call on a factor that names shared storage carries the storage key (and
// the handle's context id when the program states it). Its slot is found from the key, the
// device of the call and the context (the program's constant, else the one open context that
// declared the key on the analysis, else the handle's context read back), without reading the
// handle; the devices that run one program each hold their own slot. A refactor makes
// the storage hold the values the result keeps: one kernel assembles into the primary's copy,
// in place, the call's values in the lanes whose predicate holds and the held values elsewhere,
// comparing them with the copy, and one read brings back a word per lane (its predicate merged
// with the comparison); the lanes whose values changed are factored in place, the others
// (a lane that asks with the very values the storage holds included) keep their factor and
// generation. A solve (not transposed) that runs after a refactor of the storage in the same
// execution of the program (XLA's run id) uses the primary as it stands, whatever generation the
// handle carries; only an invalidated side is factored first, from the values the solve carries.
// Any other solve, and every transposed solve, follows the generation rule above, so a deferred
// linear program applies its own factor and reverse mode the transpose of a factor's own matrix
// (on a single slot by factoring the slot's own side again when the handle is of another
// generation). Such a solve that factors the primary again from its own values (a derivative
// solve of an older factor that runs after a later refactor of the storage, transposed or not)
// leaves the storage holding its matrix, so the storage stops being trusted until the next
// refactor of it: a later solve of the same execution follows the generation rule too.
// An unfilled storage solves to nan with the status EMPTY, and the first refactor that asks
// fills it.
//
// Static dispatch: a call whose identity is a constant of the compiled program receives it as
// attributes and finds its slot on the host, without reading an operand. A factor with
// static_context and static_lane keys its slot by them. A solve, refactor or invalidate with
// static_path (the site of the factor that produced its handle, then the site of each refactor
// the handle went through), static_context and static_lane takes the factor site's slot for
// that context and lane token, then at each refactor site the slot that site keeps for the
// context and the slot reached so far; the slot reached so far stays where the refactor holds
// it in place or returned it unchanged. The handle's generation is the one the producing call
// wrote in the same execution (its site keeps the words it wrote, below), so a call that runs
// after a later refactorization of the slot (calls that do not pass one factor from one to the
// next are not ordered) sees the generation differ; when the producing call's words are not
// kept for this execution, the slot's latest generation stands for the handle's. A side holding
// another generation is factored again from the values the call carries, as for a read handle.
// When the
// program states only that the handle is the result of one of its calls (a static_path of that
// call's site alone, with static_context and static_lane -1: the origin of the chain was a loop
// carry, a program argument or a traced context), the handler takes the handle words that call
// wrote in the same execution of the program (XLA's run id tells executions apart), exactly as
// it would read them back from the device. A call without these attributes, or whose path does
// not resolve, reads its operands.
//
// Invalidate: the slot a handle names keeps its storage, but its sides stand for no handle any
// more, so the next solve on the slot factors them again from the values it carries, and the
// next factor or refactor request on it sets a new generation as usual. A batched handle
// invalidates every lane of its batch slot.
//
// Every side with matching (general and symmetric indefinite ones) runs its own cold cuDSS
// analysis: with matching the ordering depends on the values, and a permutation kept from
// other values can be far from fill-reducing for them (measured: 150 times the LU fill on a
// matrix whose matching moved every row); a permutation passed without cuDSS's partition tree
// also leaves the symbolic phase and every later factorization and solve on a slower schedule
// (measured 3x and 1.9x at n = 90,000). Sides without matching (symmetric positive definite
// ones, and batch tiles) have a seed: the permutation and the nested-dissection partition tree
// of the first cold analysis of such a side, passed together, which give cuDSS the cold ordering
// and schedule in about a tenth of the analysis time; the first such side runs cold and keeps
// them (see CreateTile for the batch measurement). contract_set_seeding(False) makes every
// analysis cold.
//
// Pivoting: every side runs with pivot perturbation off (CUDSS_CONFIG_PIVOT_EPSILON = 0), so a
// zero pivot stays zero instead of being replaced by a small number that yields a finite but
// wrong solution. Every side except a symmetric positive definite one also runs cuDSS's
// matching (CUDSS_MATCHING_ALG_AUTO), which permutes large entries onto the diagonal and
// equilibrates the matrix, so that nonsingular operators with zero diagonal entries
// (saddle-point and constraint blocks, symmetric indefinite matrices) still factor. A
// symmetric declaration is declared to cuDSS as a general matrix, so it factors as a general LU
// with the general side's matching and fill; an analysis created with symmetric_factorization
// declares it symmetric instead, and cuDSS then factors it with its symmetry-preserving matching
// and pivoting (about half the fill on a saddle-point matrix). Symmetric positive definite sides
// run a Cholesky factorization with matching off (CUDSS_MATCHING_ALG_NONE). Every matching
// option is set explicitly, so the sides do not follow cuDSS's defaults.
//
// Status: the status of a single side is decided by rules on its values and on the diagonal of
// its factor (CUDSS_DATA_DIAG). Non-finite values make the side NONFINITE (2) (the phase ran on
// them and left a non-finite factor, which the next request replaces by a full factorization).
// Else the side is SINGULAR (1) when the diagonal holds a non-finite or a zero entry or, on a
// side that ran matching and so was equilibrated, a smallest pivot at most 16 unit roundoffs of
// its largest; SUCCESS (0) otherwise. cuDSS reports zero in CUDSS_DATA_INFO even for an
// exactly singular matrix, so the diagonal is what detects a singular factor (see
// DiagonalSingular), and CUDSS_DATA_INFO is not read. A side factored with cuDSS's symmetric
// indefinite factorization (symmetric_factorization) is the exception: its pivots may be 2 x 2
// blocks, and CUDSS_DATA_DIAG then reports only the diagonal of D in L D L^T, so a nonsingular
// matrix of blocks [[0, 1], [1, 0]] shows an all-zero diagonal while its solve is exact
// (measured with cuDSS 1.0.0.35, matching on, Bunch-Kaufman pivoting or cuDSS's default). Such a
// side keeps only the non-finite diagonal condition and takes the rest of its verdict from a
// status solve (SymmetricStatusSolve): A x = A x* for a known x*, singular when the error is
// non-finite or, in float64, more than a sixteenth of x*'s largest entry.
//
// Where and when the rules are evaluated: cuDSS hands the diagonal to the host only after
// waiting for all the work queued on the stream. A full factorization (the first of a side, a
// rebuild, a retry, every factor call) reads it at once, with one wait, and its status stands
// when the call returns. A refactorization, the request a carried factor repeats in a loop, does
// not wait: each single side keeps status words on the device (status_kernels.h), a kernel
// queued after the phase marks them NONFINITE or SUCCESS from the values, and the verdict of the
// phase is pending. A solve of a side whose verdict is pending runs, and one kernel after it
// takes the status from the device words: with finite values and finite right-hand sides a
// non-finite solution marks the side SINGULAR there (with pivot perturbation off a zero or
// non-finite pivot makes the solution infinite or nan); a side that is not SUCCESS gets nan in
// every solution entry; the call's status output is written from those words. The host settles
// a pending verdict (counted in deferred_status_reads) at the next call on the side that reads
// the host: a shared-storage refactor in its one read, a refactor with its predicate read, a
// request that factors the side again (one wait, before it queues new work), and
// contract_stats. Settling reads the device words and the diagonal in one wait, applies every
// rule above, the 16-roundoff one included, and writes the settled status back to the device
// words. A refactorization whose only fault is a smallest pivot at or below 16 unit roundoffs
// that is not zero therefore reaches its solves only once settled: until then they return the
// factor's finite solution.
//
// The matching and scaling come from the values a side was analysed with, so a full
// factorization of other values can end singular on a regular matrix: the side is then analysed
// again with the current values and factored once more before the status stands (counted as
// reanalyses). A refactorization keeps the pivot order of the factor it refactors; when its
// verdict settles singular, the call that settled it factors the side afresh from the values it
// holds (the refactor that keeps or refactors it, or the next solve). A transposed solve takes the
// status of the factor it solves with, since a matrix and its transpose are singular together.
// With pivot perturbation off no pivot is perturbed, and CUDSS_DATA_NPIVOTS is not read. A solve on a side whose settled
// status is not SUCCESS writes nan into the solution and skips the cuDSS solve. A cuDSS or CUDA
// failure is an error of the call, never a status.
//
// Batch sides: a batch side runs with matching off (CUDSS_MATCHING_ALG_NONE), so it factors every
// lane by LU (whatever the declaration) with pivot perturbation off and without matching. A stacked
// tile is one cuDSS system, whose factor diagonal (CUDSS_DATA_DIAG) holds every lane's pivots in
// the rows' original order, so one device pass over the values and diagonals of the lanes a numeric
// phase factored, read back once, gives each lane's status: a lane with non-finite values is
// NONFINITE; a lane whose diagonal holds a non-finite or a zero entry, or whose smallest pivot
// magnitude is at most the square root of the unit roundoff times its largest, is suspect. cuDSS
// reports neither CUDSS_DATA_INFO nor a diagonal per lane on a uniform batch (CUDSS_DATA_DIAG gives
// the first lane's only, measured with cuDSS 1.0.0.35), so a uniform batch settles the statuses of
// the lanes a numeric phase factored with a status solve instead: every such lane solves A_l x =
// A_l x* for a known x* (status_kernels.h), and one device pass over those lanes' values and errors
// x - x*, read back once, gives each lane's status. A lane with non-finite values is NONFINITE; a
// lane whose error is non-finite (a zero pivot) or larger than the square root of the unit roundoff
// times the largest entry of x* (an unstable or ill-conditioned factor without matching) is
// suspect. In either layout a suspect lane is factored again alone on a single side of the slot,
// configured and checked as the side of a single slot is (matching, the status rules above,
// re-analysis). That single factor (a rescue) serves the lane and gives its status until the lane's
// next numeric phase; every other lane is SUCCESS. On a uniform batch, numeric phases and solves
// run on the lanes they need only (CUDSS_DATA_UBATCH_MASK); the other lanes keep their factors. A
// stacked tile has no mask: each numeric phase and solve of the tile runs on all its lanes, and a
// lane that the request does not factor is factored again from the values its copy holds, which are
// its factor's values, so it keeps its factor's matrix, its status and its generation (counted in
// lanes_factored and in lanes_factored_unasked). A batched refactor reads its predicate vector:
// with no lane asking it returns the held handles unchanged and factors nothing; with any lane
// asking, the lanes that ask are factored (a refactorization in place, or a full factorization; a
// lane that a refactorization leaves suspect is factored afresh) from the values the call carries
// and take the new generation, while a lane that does not ask keeps its held factor, its status and
// its generation (see RefactorBatchSlot). Each lane of a batch carries its own generation, so a
// vmapped loop that freezes the lanes that have finished (their carry keeps the old handle)
// presents lanes of one batch slot at different generations; a lane behind its slot lane is
// factored again from the values the call carries (a solve) or on the next solve (a refactor in
// which it does not ask). A batched call names lanes of one batch slot, each in its place (empty
// lanes aside); a call that names one lane of a batch slot alone fails, and a vector of handles
// that names single slots runs lane by lane. cuDSS's uniform batch refuses deterministic mode and
// hybrid memory mode, so a batch of an analysis with either runs lane by lane in both layouts:
// every lane is factored and solved on a single side of its own (RunsByLane), with the same results
// and statuses as single slots.
//
// Empty factors: a handle whose slot id is negative holds no factor; empty factors carry
// (context id, -1, -1), the initial value of a loop that refactors at one site, and a
// sequential vmap may number its lanes' empty slots -2, -3, .... A solve on an empty lane
// writes nan and the status EMPTY (7) without a cuDSS call; an invalidation leaves it alone. A
// refactor factors the empty lanes that ask into the refactor site's slot for the context and
// the empty slot id (a batch slot under a native vmap, keyed by -1; for a vector of handles
// that names single slots, a slot per lane), as it factors the lanes of a held slot into the
// site's slot for that slot; an empty lane that does not ask stays empty, its handle unchanged
// and its status EMPTY. A batched handle may hold empty lanes among the lanes of one batch
// slot; they ride in the batch's tiles, whatever those factor for them, are never rescued and
// are reported EMPTY.
//
// Kept values: a refactor also takes the held factor's values (the caller's layout and dtype,
// like the call's values) and returns, aliased to the call's values, the values each lane's
// result holds: the call's for a lane that asks, the held factor's (copied in) for a lane that
// does not.
//
// Host traffic: on the CUDA platform every operand is device memory, so a call without static
// identity copies its small int32 operands (context id, slot, generation, lane token) to
// pinned host memory and waits for the stream once before any cuDSS work (counted in
// host_reads); a refactor reads its predicate the same way (counted in predicate_reads, in the
// same wait as its handle when it reads one). A value check reads its per-lane words with one
// more wait (a shared-storage refactor's one read, counted in host_reads and predicate_reads,
// which also settles the pending verdict of the storage's factor). A full factorization of a
// single side waits once for its verdict (cuDSS waits for the phase inside the diagonal read);
// a refactorization does not wait for the device, and its verdict is settled later as "Status"
// says (one wait, shared with the read of the call that settles it when it has one). A batch
// side reads its status scan back once per numeric phase (and once more when a refactorization
// leaves a lane suspect). Nothing else on the factor, refactor and solve paths waits for the
// device. Results are
// written from pinned host memory with asynchronous copies, or by a kernel: the one that fills
// a one-lane solve's solution before the solve (when the host knows the status), the value
// check of a shared storage that keeps its factor (with the kept values), the solve-status
// kernel of a pending verdict, or a launch of their own.
//
// Every handler takes a fresh XLA token as its first operand and returns one as its first
// result, and takes the factor's order word as its last operand without reading it. The
// lowering passes the call's status result (an invalidation's order word, which it returns
// aliased to the operand, untouched) as the order word of the next operation on the factor, so
// XLA runs the operations on a factor that is passed from one call to the next in that order.
//
// One mutex serialises every handler body and every module function; the Python-facing
// functions release the GIL before taking it. A slot lives on the device of the call that created
// it, and its analysis keeps device copies of its patterns on every device that holds a slot of
// it; a call that names a slot from another device than the slot's fails.
//
// Streams: a slot's cuDSS handle is bound to the stream of the slot's first call and never bound
// again, because cuDSS runs every phase of a data object on the stream bound when the object was
// analysed. A call on another stream queues its own kernels and copies on its stream and hands
// each cuDSS call over with an event each way: the handle's stream waits for the call's stream
// before the cuDSS call, and the call's stream waits for the handle's stream after it. A factor
// spread over a model group (cuDSS MG) is ordered by host waits around each phase instead (see
// "Factors over a model group").

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "assembly_kernels.h"
#include "status_kernels.h"

namespace cudss_contract {

// Result codes shared with diff_soup.linear.SolveStatus.
constexpr int32_t kSuccess = 0;
constexpr int32_t kSingular = 1;
constexpr int32_t kNonfinite = 2;
constexpr int32_t kEmpty = 7;  // the handle holds no factorization

// The slot id and generation of an empty handle, which holds no factor.
constexpr int32_t kEmptySlot = -1;

std::mutex& ContractMutex() {
    static std::mutex mutex;
    return mutex;
}

struct Stats {
    int64_t cold_analyses = 0;              // cuDSS analyses that ran the reordering
    int64_t seeded_analyses = 0;            // analyses of sides without matching seeded with
                                            // the kept permutation and partition tree
    int64_t factorizations = 0;             // numeric factorizations of a primary side
    int64_t refactorizations = 0;           // in-place refactorizations of a primary side
    int64_t rebuilds = 0;                   // factorizations a solve started for an older factor
    int64_t solves = 0;                     // cuDSS solves on a primary side
    int64_t transposed_solves = 0;          // of those, solves in transpose mode (A_bc^T)
    int64_t nonfinite = 0;                  // factorizations (lanes) skipped for non-finite values
    int64_t singular = 0;                   // factorizations (lanes) that ended singular
    int64_t released_context_errors = 0;    // calls refused because their context was released
    int64_t host_reads = 0;                 // handler calls that read their handle on the host
    int64_t zero_fills = 0;                 // solution buffers zeroed before a cuDSS solve
    int64_t perturbed_pivot_factorizations = 0;  // factorizations with perturbed pivots
    int64_t reanalyses = 0;  // sides analysed again because a stale analysis left them singular
    int64_t static_dispatches = 0;  // handler calls that found their slot from attributes
    int64_t predicate_reads = 0;    // refactor calls that read their predicate on the host
    int64_t invalidations = 0;      // invalidate calls
    int64_t batched_factorizations = 0;  // numeric phases of a batch tile
    int64_t batched_solves = 0;          // cuDSS solves of a batch tile
    int64_t lanes_factored = 0;          // lanes over batched_factorizations (those the mask asked
                                         // on a uniform batch, every lane of a stacked tile)
    int64_t lanes_factored_unasked = 0;  // of those, lanes of a stacked tile that the request did
                                         // not ask for, factored again from the values they hold
    int64_t status_solves = 0;           // cuDSS solves that settle statuses: of a batch tile
                                         // (its lanes), or of a symmetric indefinite single side
    int64_t batch_analyses = 0;  // cuDSS analyses of a batch tile, also in cold or seeded_analyses
    int64_t lane_rescues = 0;        // batch lanes factored again alone with matching
    int64_t lane_rescue_solves = 0;  // cuDSS solves of a lane on its rescue factor
    int64_t lanes_factored_alone = 0;  // batch lanes factored on a single side of their own, as
                                       // a batch in deterministic or hybrid memory mode runs
    int64_t lane_solves_alone = 0;     // cuDSS solves of such a lane
    int64_t assemblies = 0;          // value assembly kernels queued (coordinate values to CSR)
    int64_t assembled_lanes = 0;     // lanes assembled over assemblies
    int64_t value_checks = 0;        // assemblies that compared the values with a side's copy
    int64_t rebuilds_avoided = 0;    // rebuilds (single sides, or batch tiles) a value check
                                     // found unnecessary because the side held those values
    int64_t storage_value_changes = 0;  // lanes a shared-storage refactor factored because the
                                        // values it keeps differed from the storage's, unasked
    int64_t storage_asks_unchanged = 0;  // lanes whose predicate asked a shared-storage refactor
                                         // for the very values the storage held (not factored)
    int64_t pinned_blocks_allocated = 0;  // pinned host blocks the hybrid-memory pool pinned
    int64_t pinned_blocks_reused = 0;     // pinned host requests it served from a returned block
    int64_t downcast_configurations = 0;  // single sides and batch tiles configured to store a
                                          // float64 factor downcast (CUDSS_CONFIG_DOWNCAST_FACTORS)
    int64_t deferred_status_reads = 0;   // verdicts of earlier numeric phases read on the host
                                         // (the factor diagonal, one wait each, shared with the
                                         // read of the call that settles it when it has one)
};

// A slot's address at its site: (context id, lane token or held slot id).
using SiteKey = std::pair<int32_t, int64_t>;

// The device copies of an analysis's CSR pattern and assembly map on one device; null where
// nothing was uploaded.
struct DevicePattern {
    int32_t* offsets = nullptr;
    int32_t* columns = nullptr;
    int32_t* gather_offsets = nullptr;
    int32_t* gather_entries = nullptr;
    uint8_t* diagonal_add = nullptr;

    bool Resident() const { return offsets != nullptr; }
};

struct Analysis {
    int64_t id = 0;
    int32_t n = 0;
    int64_t nse = 0;
    int64_t nnz = 0;  // coordinate entries of the topology: the values per lane
    std::vector<int32_t> offsets;
    std::vector<int32_t> columns;
    // The assembly map (assembly_kernels.h): for every CSR position the coordinate entries that
    // add into it (entries in a Dirichlet row negated and shifted), and the positions of the
    // Dirichlet rows' diagonals, which receive one.
    std::vector<int32_t> gather_offsets;  // nse + 1
    std::vector<int32_t> gather_entries;  // nnz
    std::vector<uint8_t> diagonal_add;    // nse, or empty without Dirichlet rows
    int reorder_alg = -1;
    int hybrid_mode = 0;
    int64_t hybrid_device_memory_limit_mb = 0;
    int deterministic = 0;
    // cuDSS iterative-refinement steps per solve, and whether every solve zeroes its solution
    // buffer before cuDSS writes it (0 or 1). -1 (the default) picks both by the factor: none
    // for a float64 factor kept in float64, one step and the fill for a float32 or downcast
    // factor (see RefinementSteps and ZeroFills).
    int ir_nsteps = -1;
    int zero_fill = -1;
    // cuDSS's CUDSS_CONFIG_DOWNCAST_FACTORS mode for float64 factors (cudssDowncastFactorsMode_t;
    // 0 keeps them in float64; -1, the default, picks it by the slot, see DowncastMode) and
    // CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT for their factorization (0 leaves emulation off).
    int downcast_factors = -1;
    int emulation_mantissa_bits = 0;
    // Factor sides declared symmetric (not positive definite) with cuDSS's symmetric matrix type,
    // its symmetry-preserving matching and its pivoting, instead of as a general LU.
    bool symmetric_factorization = false;
    // How a native batch lays out its lanes for cuDSS (kBatchStacked or kBatchUniform; see
    // "Batch sides").
    int batch_layout = 0;
    // cuDSS MG: the CUDA ordinals of this process's model groups, `mg_group_size` consecutive
    // ordinals per group, over which every single slot of the analysis spreads its factor (see
    // "Factors over a model group"); empty, with a group size of 0, for factors on one device.
    std::vector<int> mg_devices;
    int64_t mg_group_size = 0;
    std::vector<int32_t> permutation;  // from the first cold analysis; empty until then
    // The seed of sides without matching (symmetric positive definite sides and batch tiles): the
    // permutation and the nested-dissection partition tree (CUDSS_DATA_ND_PARTITION_TREE) of the
    // first cold analysis of such a side; empty until one has run. `tree_unavailable` records that cuDSS reported no tree for it.
    std::vector<int32_t> tree_permutation;
    std::vector<char> nd_tree;
    bool tree_unavailable = false;
    int64_t tree_seeded_analyses = 0;  // seeded analyses that also took the partition tree
    std::set<int32_t> contexts;        // contexts holding a reference to the device patterns
    std::map<int, DevicePattern> patterns;  // the device copies, by device

    int64_t cold_analyses = 0;
    int64_t seeded_analyses = 0;
    int64_t seed_refusals = 0;
    int64_t lu_nnz = -1;  // nnz of L + U of the last single side's factor; -1 for none
    // Smallest over largest pivot magnitude of the last single side's factor; -1 for none, nan
    // when the diagonal held a non-finite entry.
    double pivot_ratio = -1.0;
    int64_t lu_nnz_batch = -1;  // nnz of L + U per lane of the last analysed batch tile
    double analysis_seconds_cold = 0.0;
    double analysis_seconds_seeded_last = 0.0;
    int64_t hybrid_device_memory_min = -1;
    int64_t batch_analyses = 0;
    int batch_seeded = -1;  // the last batch tile analysis: -1 none yet, 0 cold, 1 seeded
    double analysis_seconds_batch_last = 0.0;

    Analysis() = default;
    Analysis(const Analysis&) = delete;
    Analysis& operator=(const Analysis&) = delete;

    // Whether every factor of the analysis spreads over a model group (cuDSS MG).
    bool ModelGroups() const { return mg_group_size >= 2; }

    bool DeviceResident() const {
        for (const auto& entry : patterns) {
            if (entry.second.Resident()) return true;
        }
        return false;
    }

    // The device copies on `device` (null pointers where nothing was uploaded there).
    const DevicePattern& On(int device) const {
        static const DevicePattern none{};
        auto it = patterns.find(device);
        return it == patterns.end() ? none : it->second;
    }

    // Device bytes of the uploaded patterns and their assembly maps, over every device.
    int64_t PatternBytes() const {
        int64_t bytes = 0;
        for (const auto& entry : patterns) {
            if (entry.second.offsets == nullptr) continue;
            bytes += static_cast<int64_t>(n + 1 + nse) * static_cast<int64_t>(sizeof(int32_t));
            bytes += static_cast<int64_t>(nse + 1 + nnz) * static_cast<int64_t>(sizeof(int32_t));
            bytes += static_cast<int64_t>(diagonal_add.size());
        }
        return bytes;
    }

    // Free the device copies of the patterns on every device. Returns whether any were held.
    bool FreeDevice() {
        const bool held = DeviceResident();
        int current = -1;
        cudaGetDevice(&current);
        for (auto& entry : patterns) {
            cudaSetDevice(entry.first);
            DevicePattern& p = entry.second;
            void* arrays[5] = {p.offsets, p.columns, p.gather_offsets, p.gather_entries,
                               p.diagonal_add};
            for (void* array : arrays) {
                if (array != nullptr) cudaFree(array);
            }
        }
        patterns.clear();
        if (current >= 0) cudaSetDevice(current);
        return held;
    }

    ~Analysis() { FreeDevice(); }
};

// One cuDSS factor over the analysis's pattern: the factor of a single slot, or the single factor
// of one lane of a batch slot (its rescue, or every lane of a batch that runs lane by lane).
struct Side {
    cudssHandle_t handle = nullptr;  // owned by the slot
    int device = -1;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;          // CSR over the shared pattern and `values`
    cudssMatrix_t b = nullptr;          // solve-time right-hand sides, re-pointed per call
    cudssMatrix_t x = nullptr;          // solve-time solutions, re-pointed per call
    const void* bound_b = nullptr;      // the buffers `b` and `x` point at now
    void* bound_x = nullptr;
    cudssMatrix_t b_scratch = nullptr;  // (n, 1) dense operands of the analysis and
    cudssMatrix_t x_scratch = nullptr;  // factorization phases
    void* values = nullptr;             // device copy of the CSR values the factor was built from
    void* scratch_b = nullptr;
    void* scratch_x = nullptr;
    int64_t nrhs = 0;
    int64_t value_bytes = 0;
    int64_t scratch_bytes = 0;          // scratch_b and scratch_x
    int64_t permanent_device_bytes = 0;  // CUDSS_DATA_MEMORY_ESTIMATES[0], last queried
    bool factored = false;               // a factorization phase has completed on `data`
    bool analysis_current = false;       // `data` was analysed with the values now in `values`
    bool facts_known = false;            // the size facts of the factor were read once
    bool seeded = false;
    int32_t generation = 0;              // factor request the side was built for, 0 for none
    // The status the host has settled; meaningful only while `pending` is false.
    int32_t status = kSingular;
    // The factor's status words on the device (status_kernels.h), set by the kernels queued after
    // each numeric phase and each solve that the host has not settled.
    diff_soup_status::SideVerdictWords* verdict = nullptr;
    // A numeric phase ran whose verdict the host has not read yet (see "Status" above), whether
    // it was a refactorization and whether it was a full factorization on an analysis of other
    // values.
    bool pending = false;
    bool pending_refactorization = false;
    bool pending_stale = false;
    // What a settled failure asks of the side's next numeric phase: nothing, a full
    // factorization (a refactorization ended singular), or a new analysis and a full
    // factorization (a full factorization on an analysis of other values ended singular).
    int retry = 0;
    // The CUDSS_CONFIG_SOLVE_MODE the side's configuration holds: normal (A x = b over the
    // side's pattern) or transpose; set before each solve that needs the other (SetSolveMode).
    cudssSolveMode_t solve_mode = CUDSS_SOLVE_MODE_NORMAL;

    Side() = default;
    Side(const Side&) = delete;
    Side& operator=(const Side&) = delete;

    ~Side() {
        if (device >= 0) cudaSetDevice(device);
        if (A != nullptr) cudssMatrixDestroy(A);
        if (b != nullptr) cudssMatrixDestroy(b);
        if (x != nullptr) cudssMatrixDestroy(x);
        if (b_scratch != nullptr) cudssMatrixDestroy(b_scratch);
        if (x_scratch != nullptr) cudssMatrixDestroy(x_scratch);
        if (data != nullptr && handle != nullptr) cudssDataDestroy(handle, data);
        if (config != nullptr) cudssConfigDestroy(config);
        if (values != nullptr) cudaFree(values);
        if (scratch_b != nullptr) cudaFree(scratch_b);
        if (scratch_x != nullptr) cudaFree(scratch_x);
        if (verdict != nullptr) cudaFree(verdict);
    }

    int64_t ResidentBytes() const { return value_bytes + scratch_bytes + permanent_device_bytes; }
};

// Analysis::batch_layout values: the lanes of a batch tile as one block-diagonal single system
// (the default), or as a cuDSS uniform batch.
constexpr int kBatchStacked = 0;
constexpr int kBatchUniform = 1;

// Side::retry values.
constexpr int kRetryNone = 0;
constexpr int kRetryFresh = 1;
constexpr int kRetryReanalyse = 2;

// One tile of a batch side: lanes [first, first + lanes) of the slot in one data object (the
// factors of those lanes), with a configuration of its own. A uniform-batch tile's configuration
// carries the lane count (CUDSS_CONFIG_UBATCH_SIZE) and its matrices describe the tile's first
// lane; cuDSS finds lane i of the tile i * nse values, and i * n * nrhs right-hand-side and
// solution entries, further on. A stacked tile is one system of all its lanes.
struct BatchTile {
    int64_t first = 0;
    int64_t lanes = 0;
    // A stacked tile: its matrices describe all its lanes as one block-diagonal system of
    // lanes * n rows over the side's stacked pattern (CUDSS_CONFIG_UBATCH_SIZE unset, no mask).
    bool stacked = false;
    cudssHandle_t handle = nullptr;  // owned by the slot
    int device = -1;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;          // CSR over the shared pattern and the first lane's values
    cudssMatrix_t b = nullptr;          // solve-time right-hand sides, re-pointed per call
    cudssMatrix_t x = nullptr;          // solve-time solutions, re-pointed per call
    cudssMatrix_t b_scratch = nullptr;  // (n, 1) per lane over the side's scratch vectors, for
    cudssMatrix_t x_scratch = nullptr;  // the analysis, the numeric phases and the status solve
    // CUDSS_DATA_UBATCH_MASK as last set: one word per lane, 1 for a lane the next phase works
    // on. cuDSS honours a mask only on a data object that held one when it was analysed, so the
    // tile sets one (every lane) before its analysis.
    std::vector<int32_t> mask;
    int64_t nrhs = 0;
    const void* last_b = nullptr;  // the buffers the solve-time matrices point at
    const void* last_x = nullptr;
    cudssSolveMode_t solve_mode = CUDSS_SOLVE_MODE_NORMAL;  // as the configuration holds it
    int64_t permanent_device_bytes = 0;  // CUDSS_DATA_MEMORY_ESTIMATES[0], last queried
    bool facts_known = false;            // the estimates were read after a factorization once

    BatchTile() = default;
    BatchTile(const BatchTile&) = delete;
    BatchTile& operator=(const BatchTile&) = delete;

    ~BatchTile() {
        if (device >= 0) cudaSetDevice(device);
        if (A != nullptr) cudssMatrixDestroy(A);
        if (b != nullptr) cudssMatrixDestroy(b);
        if (x != nullptr) cudssMatrixDestroy(x);
        if (b_scratch != nullptr) cudssMatrixDestroy(b_scratch);
        if (x_scratch != nullptr) cudssMatrixDestroy(x_scratch);
        if (data != nullptr && handle != nullptr) cudssDataDestroy(handle, data);
        if (config != nullptr) cudssConfigDestroy(config);
    }
};

// The batched factor of a batch slot: every lane's factor of A_bc, serving both orientations
// (a solve with A_bc^T runs in cuDSS's transpose solve mode). Its lanes lie in stacked tiles
// (`stacked`) or uniform-batch tiles, by the analysis's batch_layout, except in a batch that
// runs lane by lane (`by_lane`: deterministic or hybrid memory
// mode, which cuDSS's uniform batch refuses), whose every lane has a single side of its own.
struct BatchSide {
    cudssHandle_t handle = nullptr;  // owned by the slot
    int device = -1;
    int64_t lanes = 0;
    bool by_lane = false;
    void* values = nullptr;     // every lane's CSR values, lane i at i * nse
    void* scratch_b = nullptr;  // lanes * n
    void* scratch_x = nullptr;  // lanes * n
    int64_t value_bytes = 0;
    int64_t scratch_bytes = 0;  // scratch_b and scratch_x
    // A stacked batch (kBatchStacked): the CSR pattern of the largest tile's lanes stacked
    // block-diagonally (a tile of fewer lanes uses its leading rows and entries), and, for solves
    // of several right-hand sides per lane, those right-hand sides and solutions regrouped column
    // by column (`stack_bytes` each, grown on demand). The factor diagonals of a numeric phase
    // are read into scratch_x.
    bool stacked = false;
    int32_t* stacked_offsets = nullptr;
    int32_t* stacked_columns = nullptr;
    int64_t stacked_pattern_bytes = 0;
    void* stack_b = nullptr;
    void* stack_x = nullptr;
    size_t stack_bytes = 0;
    std::vector<std::unique_ptr<BatchTile>> tiles;
    // Per lane: the single side that serves the lane (a suspect lane's rescue, created on first
    // need; every lane of a batch that runs lane by lane) and whether it serves the lane now.
    std::vector<std::unique_ptr<Side>> rescue;
    std::vector<uint8_t> rescued;
    std::vector<uint8_t> factored;  // per lane: its tile holds a factorization of the lane
    std::vector<int32_t> status;    // per lane
    // Per lane: the factor request (generation) the lane's factor was built for, 0 for none.
    std::vector<int32_t> generations;

    BatchSide() = default;
    BatchSide(const BatchSide&) = delete;
    BatchSide& operator=(const BatchSide&) = delete;

    ~BatchSide() {
        tiles.clear();
        rescue.clear();
        if (device >= 0) cudaSetDevice(device);
        if (values != nullptr) cudaFree(values);
        if (scratch_b != nullptr) cudaFree(scratch_b);
        if (scratch_x != nullptr) cudaFree(scratch_x);
        if (stacked_offsets != nullptr) cudaFree(stacked_offsets);
        if (stacked_columns != nullptr) cudaFree(stacked_columns);
        if (stack_b != nullptr) cudaFree(stack_b);
        if (stack_x != nullptr) cudaFree(stack_x);
    }

    int64_t ResidentBytes() const {
        int64_t bytes = value_bytes + scratch_bytes + stacked_pattern_bytes +
                        2 * static_cast<int64_t>(stack_bytes);
        for (const auto& tile : tiles) bytes += tile->permanent_device_bytes;
        for (const auto& side : rescue) {
            if (side) bytes += side->ResidentBytes();
        }
        return bytes;
    }
};

struct Slot {
    int32_t id = 0;        // the first of the slot's ids
    int32_t lanes = 1;     // ids [id, id + lanes) name the slot's lanes
    bool batch = false;    // built from values with a lane axis
    int64_t site = 0;      // the key of the site that created the slot
    int32_t context = -1;  // the owning SolveContext's id
    SiteKey key{};         // the slot's address at its site (unused for shared storage)
    int64_t storage = 0;   // the storage key of a shared-storage slot, 0 for a site's slot
    // The XLA execution (run id) of the latest refactor of a shared-storage slot: a solve of the
    // same execution trusts the storage to hold its factor's matrix; any other applies the
    // generation rule. -1 once a solve that applies the generation rule has factored the
    // primary again from its own values.
    int64_t trusted_run = -1;
    std::shared_ptr<Analysis> analysis;
    int device = -1;
    cudaDataType_t dtype = CUDA_R_64F;
    size_t value_size = sizeof(double);
    bool transposed = false;  // an "AT" factor: its solves apply A_bc^T to the factor of A_bc
    bool symmetric = false;
    bool spd = false;
    int32_t latest_generation = 0;  // generation of the slot's most recent factor request
    // A batch slot: each lane's generation in the handle of the slot's most recent factor
    // request (a lane that did not ask kept its own), -1 for a lane it left empty.
    std::vector<int32_t> lane_generations;
    cudssHandle_t handle = nullptr;
    std::unique_ptr<Side> primary;  // the single side (A_bc), whatever the orientation
    std::unique_ptr<BatchSide> batch_primary;  // the batch side (A_bc), whatever the orientation
    // Recorded on the stream of the slot's last use by the first call on another stream, and
    // waited for (on the device) by that call before it uses the slot's device state, so calls
    // of one slot on different streams never overlap.
    cudaEvent_t last_use = nullptr;
    // Hands a call's work over to the handle's stream and back around each cuDSS call of a call
    // on another stream (HandleEnter, HandleLeave).
    cudaEvent_t handoff = nullptr;
    // The stream of the last call that used the slot. A call on that same stream is already
    // ordered after it (a stream runs its work in order), so it neither records nor waits for
    // last_use. Meaningful once `used` is set.
    cudaStream_t last_stream = nullptr;
    bool used = false;
    // The stream the slot's cuDSS handle is bound to (cudssSetStream), once `stream_bound`: the
    // stream of the slot's first call, kept for the slot's life. cuDSS runs every phase of a data
    // object on the stream bound when the object was analysed, so the handle is never bound to
    // another one; a call on another stream orders its cuDSS work with HandleEnter and
    // HandleLeave.
    cudaStream_t bound_stream = nullptr;
    bool stream_bound = false;
    // cuDSS MG: the CUDA ordinals the slot's handle spreads its factor over, the slot's device
    // first; empty for a factor on one device.
    std::vector<int> mg_indices;

    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    ~Slot() {
        batch_primary.reset();
        primary.reset();
        if (device >= 0) cudaSetDevice(device);
        if (last_use != nullptr) cudaEventDestroy(last_use);
        if (handoff != nullptr) cudaEventDestroy(handoff);
        if (handle != nullptr) cudssDestroy(handle);
    }

    // Whether the slot's single sides run cuDSS's matching (and its equilibration): every
    // matrix type except symmetric positive definite, whose Cholesky factorization needs no
    // pivoting.
    bool Matched() const { return !spd; }

    // The matrix type the single sides declare to cuDSS: SPD for a symmetric positive definite
    // declaration; SYMMETRIC for a symmetric one only when the analysis asks for the symmetric
    // factorization; GENERAL otherwise, so a symmetric declaration factors as a general LU (with
    // the general side's matching and fill) unless asked.
    cudssMatrixType_t MatrixType() const {
        if (spd) return CUDSS_MTYPE_SPD;
        if (symmetric && analysis && analysis->symmetric_factorization) {
            return CUDSS_MTYPE_SYMMETRIC;
        }
        return CUDSS_MTYPE_GENERAL;
    }

    // Whether the single sides factor with cuDSS's symmetric indefinite factorization, whose
    // pivots may be 2 x 2 blocks: CUDSS_DATA_DIAG then holds only the diagonal of D in
    // L D L^T, which is zero for a nonsingular block [[0, a], [a, 0]] (see "Status").
    bool SymmetricIndefinite() const { return MatrixType() == CUDSS_MTYPE_SYMMETRIC; }

    // Whether the slot's factor spreads over a model group (cuDSS MG).
    bool OnModelGroup() const { return !mg_indices.empty(); }
};

// Whether the uniform batch of cuDSS would refuse the analysis's modes, so that a batch factors
// and solves its lanes one at a time on single sides (see "Batch sides").
bool RunsByLane(const Analysis& a) { return a.deterministic != 0 || a.hybrid_mode != 0; }

// The cudssDowncastFactorsMode_t of the float64 factors of ``slot``: the analysis's mode when it
// names one (0 for none), else (kDowncastDefault) KEEP_BLOCK_DIAGONAL, except on a batch slot that
// runs as native batch tiles, in either batch layout: cuDSS 1.0.0.35 refuses
// (CUDSS_STATUS_NOT_SUPPORTED) a factorization or refactorization of some lanes of a uniform batch
// (CUDSS_DATA_UBATCH_MASK) whose factors are downcast, which a batched refactor needs, so such a
// batch keeps float64 factors. Stacked tiles factor every lane and are not refused, but keep
// float64 factors by default too, so that the default precision of a batch does not depend on its
// layout. A batch that runs lane by lane factors single sides and takes the single default.
constexpr int kDowncastDefault = -1;
inline int DowncastMode(const Analysis& a, const Slot& slot) {
    if (a.downcast_factors != kDowncastDefault) return a.downcast_factors;
    if (slot.batch && !RunsByLane(a)) return 0;
    return static_cast<int>(CUDSS_DOWNCAST_FACTORS_MODE_KEEP_BLOCK_DIAGONAL);
}

// Whether ``slot`` is a float64 factor that cuDSS stores downcast to float32
// (CUDSS_CONFIG_DOWNCAST_FACTORS); a float32 factor has nothing to downcast.
inline bool Downcast(const Analysis& a, const Slot& slot) {
    return slot.dtype == CUDA_R_64F && DowncastMode(a, slot) != 0;
}

// The refinement steps a downcast factor's solves take by default. One step is the smallest
// count under which every bound of the test suite holds, the closest using 2.6e-5 of its
// tolerance (as without downcast). Measured with cuDSS 1.0.0.35 on an RTX 4070 Ti: relative
// errors of 8e-15 (against 3e-15 for the float64 factor without refinement) on 2-D
// convection-diffusion systems of 1e4 to 1e5 rows, and 3e-10 / 1e-8 normal / transposed (against
// 2e-10 / 4e-9) on the DFN iteration matrix, against 5e-8 and 1e-3 without refinement.
constexpr int kDowncastRefinementSteps = 1;

// The cuDSS iterative-refinement steps of a solve of ``slot``: none in deterministic mode,
// the analysis's count when it names one, else none for a float64 factor kept in float64,
// kDowncastRefinementSteps for a downcast one and one for a float32 factor.
// Measured with cuDSS 1.0.0 on an L40S: 10 warm solves of a 1,000,000-row float64 system without
// refinement and without the fill returned no stale output (worst relative error 4.9e-15).
inline int RefinementSteps(const Analysis& a, const Slot& slot) {
    if (a.deterministic != 0) return 0;
    if (a.ir_nsteps >= 0) return a.ir_nsteps;
    if (Downcast(a, slot)) return kDowncastRefinementSteps;
    return slot.dtype == CUDA_R_64F ? 0 : 1;
}

// Whether a solve of ``slot`` zeroes its solution buffer first: the analysis's choice when it
// makes one, else no for a float64 factor (downcast or not) and yes for a float32 one. Measured
// with cuDSS 1.0.0.35 on an RTX 4070 Ti: refined solves of downcast factors (1 to 3 steps; single
// systems and uniform batches of up to 64 lanes, 961 to 1e5 rows) from a solution buffer of NaN
// bits, or of finite garbage left there through a refactorization, were as accurate as from a
// zeroed buffer.
inline bool ZeroFills(const Analysis& a, const Slot& slot) {
    if (a.zero_fill >= 0) return a.zero_fill != 0;
    return slot.dtype != CUDA_R_64F;
}

// The refinement steps of a status solve of ``slot`` (a batch tile's, or a symmetric indefinite
// side's): none, except on a downcast factor, which takes its solves' steps. Unrefined, a
// downcast factor solves to about single precision (relative error 5e-8 on 2-D
// convection-diffusion grids, 1e-3 on the DFN iteration matrix), so its lanes would miss the
// suspect bound of the batch verdict and be factored again alone.
inline int StatusRefinementSteps(const Analysis& a, const Slot& slot) {
    return Downcast(a, slot) ? RefinementSteps(a, slot) : 0;
}

struct Site {
    int64_t id = 0;
    std::shared_ptr<Analysis> analysis;
    std::map<SiteKey, int32_t> slot_of_key;
    // (context id, held slot) -> held slot, while the site's latest call for that held slot
    // returned it unchanged.
    std::map<SiteKey, int32_t> forward_of_key;
    // The handle words (context ids, slot ids, generations; result_lanes of each) the site's
    // latest call wrote, and the XLA execution (run id) it ran in: a later call of that same
    // execution whose handle the program states to be this call's result takes them in place
    // of a device read.
    int64_t result_run_id = -1;
    int64_t result_lanes = 0;
    std::vector<int32_t> result_words;
};

// Pinned host words for one handler call: operands are read into the first words, results
// are written from the words after them. The event marks when the result copies have left the
// block.
constexpr size_t kStagingWords = 16;

struct Staging {
    int device = -1;
    int32_t* host = nullptr;
    size_t words = 0;
    cudaEvent_t done = nullptr;
    bool recorded = false;
};

// Device and pinned host scratch of one device for ScanLanes, grown on demand.
struct ScanScratch {
    void* device_words = nullptr;
    size_t device_bytes = 0;
    void* host_words = nullptr;
    size_t host_bytes = 0;
};

struct Registry {
    std::unordered_map<int64_t, std::shared_ptr<Analysis>> analyses;
    std::map<int32_t, std::unique_ptr<Slot>> slots;  // by the slot's first id
    std::unordered_map<int64_t, std::unique_ptr<Site>> sites;
    // Shared storage: the slot each (context id, device, analysis id, dtype, lanes (0 for a
    // single slot), storage key) names, and the contexts that declared each (analysis id,
    // storage key). The device keeps the slots of the devices that run one program apart.
    std::map<std::tuple<int32_t, int, int64_t, int, int64_t, int64_t>, int32_t> storage_slots;
    std::map<std::pair<int64_t, int64_t>, std::set<int32_t>> storage_contexts;
    std::unordered_set<int32_t> released_contexts;
    std::vector<std::unique_ptr<Staging>> staging;
    std::map<int, ScanScratch> scan;       // one per device, created on first use
    std::map<int, ScanScratch> flags;      // per-lane value-check words, one per device
    std::map<int, ScanScratch> settle;     // pinned host words and diagonal of a settle, per device
    int32_t check_tag = 0;                 // the tag of the latest value check (1 .. 2^28 - 1)
    int64_t next_analysis = 1;
    int32_t next_slot = 1;
    int64_t next_site = 1;  // ids of sites made by contract_site_create
    int64_t device_analyses = 0;  // analyses whose patterns are uploaded
    int64_t batch_tile_lanes = 0;  // lanes per batch tile at most; 0 for the int32 limits only
    bool seeding = true;
    Stats stats;
};

// The registry is never destroyed, so no cuDSS or CUDA object is freed during process
// teardown after the CUDA context is gone.
Registry& Reg() {
    static Registry* registry = new Registry();
    return *registry;
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

ffi::Error CudaFailed(cudaError_t err, const char* what) {
    if (err == cudaErrorMemoryAllocation) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          std::string("cuDSS factor storage: ") + what +
                              " ran out of device memory; the held factor needs more free "
                              "device memory than the device has left.");
    }
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("CUDA call ") + what +
                                                     " failed: " + cudaGetErrorString(err) +
                                                     ".");
}

#define CONTRACT_CUDA(call, what)                                  \
    do {                                                           \
        cudaError_t contract_err_ = (call);                        \
        if (contract_err_ != cudaSuccess) {                        \
            return cudss_contract::CudaFailed(contract_err_, what); \
        }                                                          \
    } while (0)

#define CONTRACT_CUDSS(call, what)                       \
    do {                                                 \
        cudssStatus_t contract_status_ = (call);         \
        if (contract_status_ != CUDSS_STATUS_SUCCESS) {  \
            return CudssCallFailed(contract_status_, what); \
        }                                                \
    } while (0)

#define CONTRACT_TRY(expr)                    \
    do {                                      \
        ffi::Error contract_error_ = (expr);  \
        if (contract_error_.failure()) {      \
            return contract_error_;           \
        }                                     \
    } while (0)

ffi::Error InvalidCall(const std::string& what) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "cuDSS held-factor call: " + what);
}

// A call that names one lane of a batch slot alone.
ffi::Error UnbatchedLaneError(const Slot& slot, int32_t lane) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "cuDSS factor slot " + std::to_string(static_cast<int64_t>(slot.id) + lane) +
                          " is lane " + std::to_string(lane) + " of a factor built under vmap " +
                          "over " + std::to_string(slot.lanes) +
                          " lanes, and this call names that lane alone; a batched factor is "
                          "solved, refactored and invalidated as a whole: solve every lane of "
                          "a batched factor under the same vmap.");
}

// A batched call whose handle lanes are neither lanes of one batch slot, each in its place, nor
// single slots only.
ffi::Error LaneOrderError(int64_t lanes) {
    return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                      "cuDSS: the " + std::to_string(lanes) +
                          " handle lanes of this call name neither every lane of one batched "
                          "factor, each in its place, nor unbatched factors only; "
                          "a call over a batched factor names all of its lanes as factor or "
                          "refactor returned them: solve every lane of a batched factor under "
                          "the same vmap.");
}

// --------------------------------------------------------------------------
// Host reads and result writes
// --------------------------------------------------------------------------

// A staging block of `device` with at least `words` words whose previous result copies have
// completed, or a new one.
ffi::Error AcquireStaging(Registry& reg, int device, size_t words, Staging** out) {
    words = std::max(words, kStagingWords);
    for (auto& block : reg.staging) {
        if (block->device != device) continue;
        if (block->recorded) {
            const cudaError_t query = cudaEventQuery(block->done);
            if (query == cudaErrorNotReady) continue;
            if (query != cudaSuccess) {
                return CudaFailed(query, "cudaEventQuery (result staging)");
            }
            block->recorded = false;
        }
        if (block->words < words) {
            int32_t* grown = nullptr;
            CONTRACT_CUDA(cudaMallocHost(reinterpret_cast<void**>(&grown), words * sizeof(int32_t)),
                          "cudaMallocHost (operand staging)");
            cudaFreeHost(block->host);
            block->host = grown;
            block->words = words;
            std::fill(block->host, block->host + words, 0);
        }
        *out = block.get();
        return ffi::Error::Success();
    }
    auto block = std::make_unique<Staging>();
    block->device = device;
    CONTRACT_CUDA(cudaMallocHost(reinterpret_cast<void**>(&block->host), words * sizeof(int32_t)),
                  "cudaMallocHost (operand staging)");
    block->words = words;
    std::fill(block->host, block->host + words, 0);
    const cudaError_t err = cudaEventCreateWithFlags(&block->done, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        cudaFreeHost(block->host);
        return CudaFailed(err, "cudaEventCreateWithFlags (result staging)");
    }
    *out = block.get();
    reg.staging.push_back(std::move(block));
    return ffi::Error::Success();
}

// A device operand copied into the staging block from word `word` on.
struct ReadSpan {
    const void* source;
    size_t bytes;
    size_t word;
};

// Copy device operands into the staging block (the words of each span are zeroed first) and
// wait for the stream once: the call's one host read. With `wait` false the copies are only
// queued, and the caller waits for them (in a wait it needs anyway).
ffi::Error ReadSpans(cudaStream_t stream, Staging& block, const ReadSpan* spans, int count,
                     bool wait = true) {
    for (int i = 0; i < count; ++i) {
        const size_t words = (spans[i].bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
        std::fill(block.host + spans[i].word, block.host + spans[i].word + words, 0);
        if (spans[i].bytes == 0) continue;
        CONTRACT_CUDA(cudaMemcpyAsync(block.host + spans[i].word, spans[i].source, spans[i].bytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync (handle operands to the host)");
    }
    if (wait) {
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (handle operands)");
    }
    return ffi::Error::Success();
}

// Write `count` int32 result arrays of `lanes` entries each from the staging block (array i
// from words [first + i * lanes, first + (i + 1) * lanes)) with asynchronous copies and mark
// the block busy until they have run.
ffi::Error WriteResultArrays(cudaStream_t stream, Staging& block, size_t first, int64_t lanes,
                             int32_t* const* targets, int count) {
    const size_t span = static_cast<size_t>(lanes);
    cudaError_t first_error = cudaSuccess;
    for (int i = 0; i < count; ++i) {
        const cudaError_t err =
            cudaMemcpyAsync(targets[i], block.host + first + static_cast<size_t>(i) * span,
                            span * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess && first_error == cudaSuccess) first_error = err;
    }
    const cudaError_t recorded = cudaEventRecord(block.done, stream);
    block.recorded = recorded == cudaSuccess;
    if (first_error != cudaSuccess) {
        return CudaFailed(first_error, "cudaMemcpyAsync (handle results to the device)");
    }
    if (recorded != cudaSuccess) {
        return CudaFailed(recorded, "cudaEventRecord (result staging)");
    }
    return ffi::Error::Success();
}

// Write the `count` result arrays of a call as WriteResultArrays does; a one-lane call writes
// its words with one kernel that carries the values as arguments instead of one copy per array,
// and leaves the staging block free at once. The last array is the status: a lane whose
// `status_sources` entry (when given, one per lane) is not null takes the device word it points
// at, a status the device holds, instead of the staging block's word.
ffi::Error WriteResults(cudaStream_t stream, Staging& block, size_t first, int64_t lanes,
                        int32_t* const* targets, int count,
                        const int32_t* const* status_sources = nullptr) {
    if (lanes != 1 || count > diff_soup_assembly::ResultWords::kMax) {
        CONTRACT_TRY(WriteResultArrays(stream, block, first, lanes, targets, count));
        if (status_sources == nullptr) return ffi::Error::Success();
        for (int64_t l = 0; l < lanes; ++l) {
            if (status_sources[l] == nullptr) continue;
            CONTRACT_CUDA(cudaMemcpyAsync(targets[count - 1] + l, status_sources[l],
                                          sizeof(int32_t), cudaMemcpyDeviceToDevice, stream),
                          "cudaMemcpyAsync (status from the device)");
        }
        return ffi::Error::Success();
    }
    diff_soup_assembly::ResultWords words;
    words.count = count;
    for (int i = 0; i < count; ++i) {
        words.targets[i] = targets[i];
        words.values[i] = block.host[first + static_cast<size_t>(i)];
    }
    if (status_sources != nullptr) words.sources[count - 1] = status_sources[0];
    CONTRACT_CUDA(diff_soup_assembly::QueueResultWords(stream, words), "result words kernel");
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Call attributes
// --------------------------------------------------------------------------

enum class CallKind { kFactor, kSolve, kRefactor, kInvalidate };

// The attributes of a held-factor call. The static identity is optional: without it the call
// reads its operands.
struct CallAttrs {
    int64_t site = 0;  // the key of the call's site (factor, refactor)
    bool transposed = false;
    bool symmetric = false;
    bool spd = false;
    bool transpose = false;  // a solve with the transpose of the handle's matrix
    bool adopt = false;  // refactor in place in the held slot, whichever site owns it
    int64_t analysis_id = 0;
    int64_t static_context = -1;  // -1: read the context id
    int64_t static_lane = -1;     // -1: read the lane token
    std::vector<int64_t> static_path;  // empty: read the handle
    int64_t storage = 0;           // shared-storage key of the factor, 0 for none
    int64_t storage_context = -1;  // the factor's context id when it is a program constant

    // Whether the context id and lane token are constants of the program.
    bool StaticKey() const {
        return static_context >= 0 && static_context <= std::numeric_limits<int32_t>::max() &&
               static_lane != -1 && static_lane >= std::numeric_limits<int32_t>::min() &&
               static_lane <= std::numeric_limits<int32_t>::max();
    }

    // Whether the handle resolves from the attributes.
    bool StaticPath() const { return StaticKey() && !static_path.empty(); }

    // Whether the program states only that the handle is the result of the call at the last
    // site of the path (in the same execution), with the key of the origin unknown.
    bool RunLocalPath() const { return !static_path.empty(); }
};

ffi::Error AttrMissing(const char* name, const char* type) {
    return InvalidCall(std::string("the attribute '") + name + "' must be " + type +
                       "; the call lacks it or passes another type.");
}

ffi::Error IntAttr(const ffi::Dictionary& attrs, const char* name, bool required,
                   int64_t* out) {
    if (!attrs.contains(name)) {
        return required ? AttrMissing(name, "an integer scalar") : ffi::Error::Success();
    }
    if (attrs.contains<int64_t>(name)) {
        *out = attrs.get<int64_t>(name).value();
        return ffi::Error::Success();
    }
    if (attrs.contains<int32_t>(name)) {
        *out = attrs.get<int32_t>(name).value();
        return ffi::Error::Success();
    }
    return AttrMissing(name, "an integer scalar");
}

ffi::Error BoolAttr(const ffi::Dictionary& attrs, const char* name, bool* out) {
    if (!attrs.contains<bool>(name)) return AttrMissing(name, "a boolean scalar");
    *out = attrs.get<bool>(name).value();
    return ffi::Error::Success();
}

ffi::Error PathAttr(const ffi::Dictionary& attrs, const char* name, std::vector<int64_t>* out) {
    out->clear();
    if (!attrs.contains(name)) return ffi::Error::Success();
    {
        ffi::ErrorOr<ffi::Span<const int64_t>> path = attrs.get<ffi::Span<const int64_t>>(name);
        if (path.has_value()) {
            out->assign(path->begin(), path->end());
            return ffi::Error::Success();
        }
    }
    {
        ffi::ErrorOr<ffi::Span<const int32_t>> path = attrs.get<ffi::Span<const int32_t>>(name);
        if (path.has_value()) {
            out->assign(path->begin(), path->end());
            return ffi::Error::Success();
        }
    }
    return AttrMissing(name, "an integer array");
}

ffi::Error DecodeAttrs(const ffi::Dictionary& attrs, CallKind kind, CallAttrs* out) {
    CONTRACT_TRY(IntAttr(attrs, "analysis_id", true, &out->analysis_id));
    if (kind == CallKind::kFactor || kind == CallKind::kRefactor) {
        CONTRACT_TRY(IntAttr(attrs, "site", true, &out->site));
        if (out->site <= 0) {
            return InvalidCall("the site attribute " + std::to_string(out->site) +
                               " is not a site key; a site key is a positive integer.");
        }
        CONTRACT_TRY(BoolAttr(attrs, "transposed", &out->transposed));
        CONTRACT_TRY(BoolAttr(attrs, "symmetric", &out->symmetric));
        CONTRACT_TRY(BoolAttr(attrs, "spd", &out->spd));
    }
    if (kind == CallKind::kSolve) {
        CONTRACT_TRY(BoolAttr(attrs, "transpose", &out->transpose));
    }
    if (kind == CallKind::kRefactor) {
        CONTRACT_TRY(BoolAttr(attrs, "adopt", &out->adopt));
    }
    CONTRACT_TRY(IntAttr(attrs, "static_context", false, &out->static_context));
    CONTRACT_TRY(IntAttr(attrs, "static_lane", false, &out->static_lane));
    if (kind != CallKind::kFactor) {
        CONTRACT_TRY(PathAttr(attrs, "static_path", &out->static_path));
        CONTRACT_TRY(IntAttr(attrs, "storage", false, &out->storage));
        CONTRACT_TRY(IntAttr(attrs, "storage_context", false, &out->storage_context));
    }
    return ffi::Error::Success();
}

// The lane count of a call's handle leaves: 1 for scalar leaves, the length of vector leaves.
// The three leaves share one shape, and the values operand (rank `values_rank`, or -1 when the
// call has none) carries one more axis than they do.
ffi::Error HandleLanes(ffi::Span<const int64_t> context, ffi::Span<const int64_t> slot,
                       ffi::Span<const int64_t> generation, int64_t values_rank,
                       int64_t* lanes) {
    const size_t rank = slot.size();
    const bool shapes_agree = context.size() == rank && generation.size() == rank &&
                              (rank == 0 || (context[0] == slot[0] && generation[0] == slot[0]));
    if (rank > 1 || !shapes_agree ||
        (values_rank >= 0 && values_rank != static_cast<int64_t>(rank) + 1)) {
        return InvalidCall("the handle leaves must be three scalars, or three vectors of one "
                           "length under a native vmap, with values of one more axis; got "
                           "leaves of ranks " +
                           std::to_string(context.size()) + ", " + std::to_string(rank) + ", " +
                           std::to_string(generation.size()) + " and values of rank " +
                           std::to_string(values_rank) + ".");
    }
    *lanes = rank == 1 ? slot[0] : 1;
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Registry lookups and checks
// --------------------------------------------------------------------------

// The slot whose first id is `id`.
Slot* FindSlot(Registry& reg, int32_t id) {
    auto it = reg.slots.find(id);
    return it == reg.slots.end() ? nullptr : it->second.get();
}

// The slot one of whose ids is `id`, and the lane `id` names in it.
Slot* FindSlotOf(Registry& reg, int32_t id, int32_t* lane) {
    auto it = reg.slots.upper_bound(id);
    if (it == reg.slots.begin()) return nullptr;
    --it;
    Slot* slot = it->second.get();
    const int64_t offset = static_cast<int64_t>(id) - slot->id;
    if (offset < 0 || offset >= slot->lanes) return nullptr;
    *lane = static_cast<int32_t>(offset);
    return slot;
}

Site* FindSite(Registry& reg, int64_t id) {
    auto it = reg.sites.find(id);
    return it == reg.sites.end() ? nullptr : it->second.get();
}

// Refuse a context that cannot own factors: none (a negative id) or one that was released.
ffi::Error CheckContext(Registry& reg, int32_t context) {
    if (context < 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS keeps its factors in device storage that a SolveContext owns, "
                          "but this call carries no context (context id " +
                              std::to_string(context) +
                              "); pass ctx=diff_soup.linear.SolveContext() to factor().");
    }
    if (reg.released_contexts.count(context) != 0) {
        ++reg.stats.released_context_errors;
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "cuDSS: the SolveContext with context_id " + std::to_string(context) +
                              " was released, so the factors built in it no longer exist; "
                              "build the factor again in an open SolveContext.");
    }
    return ffi::Error::Success();
}

// The slot a handle (context, slot id) names and the lane the id names in it, after checking
// that its context is open, the slot exists and belongs to that context.
ffi::Error FindHeld(Registry& reg, int32_t context, int32_t id, Slot** held, int32_t* lane) {
    CONTRACT_TRY(CheckContext(reg, context));
    Slot* slot = FindSlotOf(reg, id, lane);
    if (slot == nullptr) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "cuDSS factor slot " + std::to_string(id) +
                              " does not exist in the SolveContext with context_id " +
                              std::to_string(context) +
                              "; pass a handle that diff_soup.linear.factor or refactor "
                              "returned in an open SolveContext of this process.");
    }
    if (slot->context != context) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "cuDSS factor slot " + std::to_string(id) +
                              " belongs to the SolveContext with context_id " +
                              std::to_string(slot->context) +
                              ", but the handle names context_id " + std::to_string(context) +
                              "; pass the handle unchanged as factor or refactor returned it.");
    }
    *held = slot;
    return ffi::Error::Success();
}

// Refuse a slot built for another analysis than the call's.
ffi::Error CheckAnalysis(const Slot& slot, int64_t analysis_id) {
    if (slot.analysis->id != analysis_id) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS: the handle names a factor slot built for a different sparsity "
                          "pattern (slot " +
                              std::to_string(slot.id) + " belongs to analysis " +
                              std::to_string(slot.analysis->id) + "; this call's analysis is " +
                              std::to_string(analysis_id) +
                              "); solve and refactor a factor with the Analysis it was built "
                              "with.");
    }
    return ffi::Error::Success();
}

// The site of a lowered factor or refactor call: the site under `key`, created on the first call
// that carries the key, on the call's analysis. A site key is a deterministic hash of the lowered
// call, so two lowerings of one program share their sites, or an id contract_site_create made.
ffi::Error SiteFor(Registry& reg, int64_t key, int64_t analysis_id, Site** out) {
    Site* site = FindSite(reg, key);
    if (site == nullptr) {
        auto found = reg.analyses.find(analysis_id);
        if (found == reg.analyses.end()) {
            return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                              "cuDSS analysis " + std::to_string(analysis_id) +
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
                          "cuDSS factor site " + std::to_string(key) +
                              " was created for analysis " +
                              std::to_string(site->analysis->id) +
                              ", but the call names analysis " + std::to_string(analysis_id) +
                              "; a site serves the analysis of the call that created it.");
    }
    *out = site;
    return ffi::Error::Success();
}

ffi::Error CheckDtype(const Slot& slot, cudaDataType_t dtype) {
    if (slot.dtype != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS factor slot " + std::to_string(slot.id) +
                              " holds a factor of another precision than this call's values; "
                              "solve and refactor a factor at the precision it was built with.");
    }
    return ffi::Error::Success();
}

ffi::Error CheckDevice(const Slot& slot, int device) {
    if (slot.device != device) {
        return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                          "cuDSS factor slot " + std::to_string(slot.id) + " lives on device " +
                              std::to_string(slot.device) + ", but this call runs on device " +
                              std::to_string(device) +
                              "; a factor is used on the device it was built on. Factor it on "
                              "this device, or start it from empty_factor(..., storage=...), "
                              "whose shared storage keeps one factor per device.");
    }
    return ffi::Error::Success();
}

struct SlotSpec {
    bool transposed = false;
    bool symmetric = false;
    bool spd = false;
    cudaDataType_t dtype = CUDA_R_64F;
    size_t value_size = sizeof(double);
};

ffi::Error CheckSpec(const Slot& slot, const SlotSpec& spec) {
    if (slot.transposed != spec.transposed || slot.symmetric != spec.symmetric ||
        slot.spd != spec.spd || slot.dtype != spec.dtype) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "cuDSS factor slot " + std::to_string(slot.id) +
                              " was created with another orientation, symmetry or precision "
                              "than this call declares; a factor site is lowered with one set "
                              "of these attributes.");
    }
    return ffi::Error::Success();
}

// Refuse an in-place refactorization of a slot whose orientation, structure declaration or
// precision differ from the call's. For a slot the call adopts from another site that is a
// caller's mistake and is named as such; for the site's own slot it cannot happen (a site is
// lowered with one set of these attributes).
ffi::Error CheckInPlace(const Slot& slot, const SlotSpec& spec, bool adopted) {
    if (!adopted) return CheckSpec(slot, spec);
    if (slot.transposed != spec.transposed || slot.symmetric != spec.symmetric ||
        slot.spd != spec.spd || slot.dtype != spec.dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS: refactor(adopt=True) refactors in place in the storage the held "
                          "factor names, so the held factor must have this refactor's orientation, "
                          "structure declaration and precision; factor slot " +
                              std::to_string(slot.id) +
                              " was built with others. Pass a factor of the same analysis, "
                              "orientation and precision, or adopt=False.");
    }
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Pinned host memory of hybrid-memory factors
// --------------------------------------------------------------------------

// Under hybrid memory cuDSS keeps the host part of a factor in pinned host memory, which it takes
// from the handle's pinned memory handler once per data object (at its first factorization) and
// returns when the data object is destroyed. The handles of every slot of a hybrid analysis share
// one pooled handler: a returned block is kept and handed to the next request of at most its size
// and more than half of it, so a new side, a re-analysis (a fresh data object) or a lane of a
// batch that runs lane by lane takes a block another data object returned instead of pinning
// fresh memory. Idle blocks above kPinnedPoolIdleBytes in all are freed, largest first. Measured
// with cuDSS 1.0.0.35 on an RTX 4070 Ti, 64,000 rows held at the hybrid device memory minimum
// (one 235 MB block per data object): the first factorization of each new data object took 571 ms
// with cuDSS's own allocation and 518 ms from the pool (later factorizations 530 and 516 ms), and
// destroying a data object 20.5 ms and 4.4 ms.
constexpr size_t kPinnedPoolIdleBytes = size_t{1} << 30;

struct PinnedPool {
    std::mutex mutex;
    std::multimap<size_t, void*> idle;        // idle blocks by size
    std::unordered_map<void*, size_t> sizes;  // every block the pool pinned and holds, by address
    size_t idle_bytes = 0;
};

// The pool is never destroyed, like the registry.
PinnedPool& Pinned() {
    static PinnedPool* pool = new PinnedPool();
    return *pool;
}

// cudssPinnedMemHandler_t callbacks; 0 on success. Called inside cuDSS calls, which run under the
// contract mutex, so the counters of the registry may be updated here.
int PinnedAlloc(void* ctx, void** ptr, size_t size, cudaStream_t /*stream*/) {
    auto& pool = *static_cast<PinnedPool*>(ctx);
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto it = pool.idle.lower_bound(size);
    if (it != pool.idle.end() && it->first / 2 < std::max<size_t>(size, 1)) {
        *ptr = it->second;
        pool.idle_bytes -= it->first;
        pool.idle.erase(it);
        ++Reg().stats.pinned_blocks_reused;
        return 0;
    }
    void* block = nullptr;
    // Portable, so a block that a slot on one device returned serves a slot on another.
    if (cudaHostAlloc(&block, std::max<size_t>(size, 1), cudaHostAllocPortable) != cudaSuccess) {
        *ptr = nullptr;
        return 1;
    }
    pool.sizes[block] = std::max<size_t>(size, 1);
    ++Reg().stats.pinned_blocks_allocated;
    *ptr = block;
    return 0;
}

int PinnedFree(void* ctx, void* ptr, size_t /*size*/, cudaStream_t stream) {
    if (ptr == nullptr) return 0;
    auto& pool = *static_cast<PinnedPool*>(ctx);
    // Work queued on the stream may still read the block; it is handed out again only after it.
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 1;
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto known = pool.sizes.find(ptr);
    if (known == pool.sizes.end()) return 1;
    const size_t size = known->second;
    if (size > kPinnedPoolIdleBytes) {
        pool.sizes.erase(known);
        return cudaFreeHost(ptr) == cudaSuccess ? 0 : 1;
    }
    pool.idle.emplace(size, ptr);
    pool.idle_bytes += size;
    while (pool.idle_bytes > kPinnedPoolIdleBytes) {
        auto largest = std::prev(pool.idle.end());
        pool.idle_bytes -= largest->first;
        pool.sizes.erase(largest->second);
        cudaFreeHost(largest->second);
        pool.idle.erase(largest);
    }
    return 0;
}

// Give `handle` the pooled pinned memory handler.
ffi::Error UsePinnedPool(cudssHandle_t handle) {
    cudssPinnedMemHandler_t handler{};
    handler.ctx = &Pinned();
    handler.pinned_alloc = PinnedAlloc;
    handler.pinned_free = PinnedFree;
    std::strncpy(handler.name, "diff_soup pinned pool", CUDSS_ALLOCATOR_NAME_LEN - 1);
    CONTRACT_CUDSS(cudssSetPinnedMemHandler(handle, &handler),
                   "cudssSetPinnedMemHandler for a held factor");
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Factors over a model group (cuDSS MG)
// --------------------------------------------------------------------------

// An analysis with model groups (contract_analysis_set_model_groups) spreads the factor of each
// of its single slots over the CUDA devices of one model group of this process with cuDSS MG: the
// group that holds the device of the slot's first call, that device first. The slot's handle is
// a multi-device handle (cudssCreateMg, after peer access is enabled across the group), and every
// configuration of the slot names the group (CUDSS_CONFIG_DEVICE_COUNT and
// CUDSS_CONFIG_DEVICE_INDICES) before its analysis. The pattern, the value copy, the right-hand
// sides, the solutions and the status words stay on the slot's device, as on one device, and the
// slot is held, refactored in place and solved like any other single slot. cuDSS MG runs a phase
// on streams of its own that are not ordered after the call's stream, so the host waits for the
// call's stream (and the handle's) before each phase and for the handle's stream after it
// (ExecuteOnModelGroup). The phase has then completed, so the verdict of a refactorization is
// settled at once instead of deferred (FactorSide), with one read of the factor's diagonal
// (CUDSS_DATA_DIAG, which cuDSS reports on a multi-device handle). Analyses of these sides run cold
// (AnalyzeSide: no seed), and a batch slot is refused, since one factorization spans the group.

// The cuDSS handle of the new slot `slot` (its analysis, device and lane count set): a
// multi-device handle over the model group that holds the slot's device when the analysis has
// model groups (the group recorded in the slot), a one-device handle otherwise.
ffi::Error CreateSlotHandle(Slot& slot) {
    const Analysis& a = *slot.analysis;
    if (!a.ModelGroups()) {
        CONTRACT_CUDSS(cudssCreate(&slot.handle), "cudssCreate for a held factor");
        return ffi::Error::Success();
    }
    if (slot.batch) {
        return ffi::Error(
            ffi::ErrorCode::kInvalidArgument,
            "cuDSS on a model axis runs one factorization over the devices of a model group, so "
            "a factor takes the values of one matrix, of shape (nnz,); got values with a lane "
            "axis of " + std::to_string(slot.lanes) +
                " (a jax.vmap over the operator values). Batch the right-hand sides only, or "
                "factor each matrix in a call of its own.");
    }
    const std::string refusal =
        ModelGroupOf(a.mg_devices, a.mg_group_size, slot.device, &slot.mg_indices);
    if (!refusal.empty()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, refusal);
    CONTRACT_TRY(EnableMgPeerAccess(slot.mg_indices, slot.device));
    CONTRACT_CUDSS(cudssCreateMg(&slot.handle, static_cast<int>(slot.mg_indices.size()),
                                 slot.mg_indices.data()),
                   "cudssCreateMg for a held factor");
    return ffi::Error::Success();
}

// Name the slot's model group on a new configuration of `slot`, before its analysis; nothing for
// a slot on one device.
ffi::Error ConfigureModelGroup(const Slot& slot, cudssConfig_t config) {
    if (!slot.OnModelGroup()) return ffi::Error::Success();
    int count = static_cast<int>(slot.mg_indices.size());
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_DEVICE_COUNT, &count, sizeof(count)),
                   "cudssConfigSet device_count of a held factor");
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_DEVICE_INDICES,
                                  const_cast<int*>(slot.mg_indices.data()),
                                  static_cast<size_t>(count) * sizeof(int)),
                   "cudssConfigSet device_indices of a held factor");
    return ffi::Error::Success();
}

// One cudssExecute on the multi-device handle of `slot` from a call on `stream`: the host waits
// for the call's stream and the handle's before the phase, so the phase reads the values and
// right-hand sides the call queued, and for the handle's stream after it, so the work the call
// queues next reads what the phase wrote. `status` receives the phase's status.
ffi::Error ExecuteOnModelGroup(const Slot& slot, cudaStream_t stream, cudssPhase_t phase,
                               cudssConfig_t config, cudssData_t data, cudssMatrix_t A,
                               cudssMatrix_t x, cudssMatrix_t b, cudssStatus_t* status) {
    const bool other = slot.stream_bound && slot.bound_stream != stream;
    CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (before an MG phase)");
    if (other) {
        CONTRACT_CUDA(cudaStreamSynchronize(slot.bound_stream),
                      "cudaStreamSynchronize (before an MG phase)");
    }
    *status = cudssExecute(slot.handle, phase, config, data, A, x, b);
    CONTRACT_CUDA(cudaStreamSynchronize(other ? slot.bound_stream : stream),
                  "cudaStreamSynchronize (after an MG phase)");
    return ffi::Error::Success();
}

// The slot of `site` for `key` in `context`, created on first use (with its cuDSS handle) and
// counted as a reference of its context on the analysis. A batch slot (`batch`) of `lanes`
// lanes reserves `lanes` consecutive slot ids.
ffi::Error SiteSlot(Registry& reg, Site& site, int32_t context, int64_t key,
                    const SlotSpec& spec, int device, bool batch, int64_t lanes, Slot** out) {
    const SiteKey address{context, key};
    auto it = site.slot_of_key.find(address);
    if (it != site.slot_of_key.end()) {
        if (Slot* slot = FindSlot(reg, it->second)) {
            CONTRACT_TRY(CheckSpec(*slot, spec));
            if (slot->batch != batch || slot->lanes != lanes) {
                return ffi::Error(ffi::ErrorCode::kInternal,
                                  "cuDSS factor site " + std::to_string(site.id) +
                                      " holds a slot of " + std::to_string(slot->lanes) +
                                      " lanes for this key, but this call carries " +
                                      std::to_string(lanes) +
                                      "; a factor site is lowered with one lane count.");
            }
            *out = slot;
            return ffi::Error::Success();
        }
    }
    if (lanes < 1 || reg.next_slot > std::numeric_limits<int32_t>::max() - lanes) {
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          "cuDSS slot identifiers are exhausted in this process.");
    }
    auto slot = std::make_unique<Slot>();
    slot->site = site.id;
    slot->context = context;
    slot->key = address;
    slot->analysis = site.analysis;
    slot->device = device;
    slot->dtype = spec.dtype;
    slot->value_size = spec.value_size;
    slot->transposed = spec.transposed;
    slot->symmetric = spec.symmetric;
    slot->spd = spec.spd;
    slot->batch = batch;
    slot->lanes = static_cast<int32_t>(lanes);
    CONTRACT_TRY(CreateSlotHandle(*slot));
    if (slot->analysis->hybrid_mode != 0) CONTRACT_TRY(UsePinnedPool(slot->handle));
    CONTRACT_CUDA(cudaEventCreateWithFlags(&slot->last_use, cudaEventDisableTiming),
                  "cudaEventCreateWithFlags (slot order)");
    CONTRACT_CUDA(cudaEventCreateWithFlags(&slot->handoff, cudaEventDisableTiming),
                  "cudaEventCreateWithFlags (factor stream handoff)");
    slot->id = reg.next_slot;
    reg.next_slot += static_cast<int32_t>(lanes);
    Slot* raw = slot.get();
    reg.slots.emplace(raw->id, std::move(slot));
    site.slot_of_key[address] = raw->id;
    raw->analysis->contexts.insert(context);
    *out = raw;
    return ffi::Error::Success();
}

// The shared-storage slot of `storage` in `context` on `device` for factors of `analysis_id` at
// `dtype` with `lanes` lanes (0 for a single slot), or null when none was filled yet.
Slot* FindStorageSlot(Registry& reg, int32_t context, int device, int64_t analysis_id,
                      cudaDataType_t dtype, int64_t lanes, int64_t storage) {
    auto it = reg.storage_slots.find(
        {context, device, analysis_id, static_cast<int>(dtype), lanes, storage});
    return it == reg.storage_slots.end() ? nullptr : FindSlot(reg, it->second);
}

// Refuse a shared-storage slot built with another orientation or structure declaration than the
// call's: one storage key names one factor per analysis, precision and lane count.
ffi::Error CheckStorageSpec(const Slot& slot, const SlotSpec& spec) {
    if (slot.transposed != spec.transposed || slot.symmetric != spec.symmetric ||
        slot.spd != spec.spd) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS: the shared storage this factor names (empty_factor(..., "
                          "storage=...)) holds a factor of another orientation or structure "
                          "declaration (symmetric, spd) than this call's; one storage name "
                          "serves one orientation and declaration per analysis, precision and "
                          "lane count. Give factors of another orientation or declaration a "
                          "storage name of their own.");
    }
    return ffi::Error::Success();
}

// The shared-storage slot of `storage` in `context` on `device`, created on first use as SiteSlot
// creates a site's slot (recording `site` as its creator), and kept in the registry's storage map
// instead of a site's.
ffi::Error StorageSlot(Registry& reg, Site& site, int32_t context, int64_t storage,
                       const SlotSpec& spec, int device, bool batch, int64_t lanes, Slot** out) {
    const int64_t key_lanes = batch ? lanes : 0;
    if (Slot* found = FindStorageSlot(reg, context, device, site.analysis->id, spec.dtype,
                                      key_lanes, storage)) {
        CONTRACT_TRY(CheckStorageSpec(*found, spec));
        *out = found;
        return ffi::Error::Success();
    }
    // A key no site uses for a slot: storage slots are found through the storage map.
    const int64_t unused = std::numeric_limits<int64_t>::min();
    Slot* slot = nullptr;
    CONTRACT_TRY(SiteSlot(reg, site, context, unused, spec, device, batch, lanes, &slot));
    site.slot_of_key.erase(SiteKey{context, unused});
    slot->storage = storage;
    reg.storage_slots[{context, device, site.analysis->id, static_cast<int>(spec.dtype),
                       key_lanes, storage}] = slot->id;
    *out = slot;
    return ffi::Error::Success();
}

// The context of a shared-storage call without reading its handle: the context id the program
// states as a constant, else the one open context that declared the storage key on the call's
// analysis; -1 when neither holds (the call then reads the handle's context id).
int32_t StorageContext(const Registry& reg, const CallAttrs& at) {
    if (at.storage_context >= 0 && at.storage_context <= std::numeric_limits<int32_t>::max()) {
        return static_cast<int32_t>(at.storage_context);
    }
    auto it = reg.storage_contexts.find({at.analysis_id, at.storage});
    if (it != reg.storage_contexts.end() && it->second.size() == 1) {
        const int32_t context = *it->second.begin();
        if (reg.released_contexts.count(context) == 0) return context;
    }
    return -1;
}

// Bind the slot's cuDSS handle to the call's stream on the slot's first call, before any of its
// data objects is analysed, and never again: cuDSS runs every phase of a data object on the
// stream bound at its analysis, so a later cudssSetStream would not move the work and would only
// hide where it runs. Then order the call after the slot's last use: a call on the stream of that
// use is already ordered after it; a call on another stream records the slot's event on the
// stream of the last use, which marks everything queued there so far (the last use included),
// and waits for it, a wait the device performs, so the host does not wait. The cuDSS calls of a
// call on another stream than the handle's are ordered with HandleEnter and HandleLeave.
ffi::Error BindSlot(Slot& slot, cudaStream_t stream) {
    if (!slot.stream_bound) {
        CONTRACT_CUDSS(cudssSetStream(slot.handle, stream), "cudssSetStream of a held factor");
        slot.bound_stream = stream;
        slot.stream_bound = true;
    }
    if (slot.used && slot.last_stream != stream) {
        CONTRACT_CUDA(cudaEventRecord(slot.last_use, slot.last_stream),
                      "cudaEventRecord (slot order)");
        CONTRACT_CUDA(cudaStreamWaitEvent(stream, slot.last_use, 0),
                      "cudaStreamWaitEvent (slot order)");
    }
    return ffi::Error::Success();
}

// Remember the stream of the call's work on the slot's device state; the next call on another
// stream orders itself after that work (BindSlot).
ffi::Error MarkSlotUse(Slot& slot, cudaStream_t stream) {
    slot.last_stream = stream;
    slot.used = true;
    return ffi::Error::Success();
}

// Before a cuDSS call on the slot's handle from a call on `stream`: when the handle is bound to
// another stream, that stream waits (on the device) for everything `stream` queued so far, the
// values and right-hand sides the cuDSS call reads included. Nothing happens on the handle's own
// stream.
ffi::Error HandleEnter(const Slot& slot, cudaStream_t stream) {
    if (!slot.stream_bound || stream == slot.bound_stream) return ffi::Error::Success();
    CONTRACT_CUDA(cudaEventRecord(slot.handoff, stream),
                  "cudaEventRecord (call stream to factor stream)");
    CONTRACT_CUDA(cudaStreamWaitEvent(slot.bound_stream, slot.handoff, 0),
                  "cudaStreamWaitEvent (call stream to factor stream)");
    return ffi::Error::Success();
}

// After a cuDSS call on the slot's handle from a call on `stream`: when the handle is bound to
// another stream, `stream` waits (on the device) for the cuDSS work, so the kernels and copies it
// queues next read the factor, its diagonal and the solution after cuDSS wrote them.
ffi::Error HandleLeave(const Slot& slot, cudaStream_t stream) {
    if (!slot.stream_bound || stream == slot.bound_stream) return ffi::Error::Success();
    CONTRACT_CUDA(cudaEventRecord(slot.handoff, slot.bound_stream),
                  "cudaEventRecord (factor stream to call stream)");
    CONTRACT_CUDA(cudaStreamWaitEvent(stream, slot.handoff, 0),
                  "cudaStreamWaitEvent (factor stream to call stream)");
    return ffi::Error::Success();
}

// Make `side`'s configuration hold solve mode `mode` (CUDSS_CONFIG_SOLVE_MODE): transpose for a
// solve with the transpose of the side's matrix, normal for every other phase. cuDSS takes a
// change of the mode on the configuration of a factored data object (measured with cuDSS
// 1.0.0.35: the next solve runs in the new mode, as accurately as the normal one).
ffi::Error SetSolveMode(Side& side, cudssSolveMode_t mode) {
    if (side.solve_mode == mode) return ffi::Error::Success();
    CONTRACT_CUDSS(cudssConfigSet(side.config, CUDSS_CONFIG_SOLVE_MODE, &mode, sizeof(mode)),
                   "cudssConfigSet solve_mode of a held factor");
    side.solve_mode = mode;
    return ffi::Error::Success();
}

// One cudssExecute on the slot's handle from a call on `stream`, ordered with HandleEnter and
// HandleLeave. `status` receives the phase's status; the error is a failure of the ordering.
ffi::Error ExecuteOnHandle(const Slot& slot, cudaStream_t stream, cudssPhase_t phase,
                           cudssConfig_t config, cudssData_t data, cudssMatrix_t A,
                           cudssMatrix_t x, cudssMatrix_t b, cudssStatus_t* status) {
    if (slot.OnModelGroup()) {
        return ExecuteOnModelGroup(slot, stream, phase, config, data, A, x, b, status);
    }
    CONTRACT_TRY(HandleEnter(slot, stream));
    *status = cudssExecute(slot.handle, phase, config, data, A, x, b);
    return HandleLeave(slot, stream);
}

int32_t NextGeneration(Slot& slot) {
    slot.latest_generation = slot.latest_generation == std::numeric_limits<int32_t>::max()
                                 ? 1
                                 : slot.latest_generation + 1;
    return slot.latest_generation;
}

// The slot a static path names in `context` for lane token `lane`: the first site's slot for
// (context, lane), then at each later site the slot it keeps for (context, the slot reached so
// far), or the slot reached so far when that site refactors it in place or returned it
// unchanged. Null when the path does not resolve; the call then reads its handle.
Slot* ResolveStaticPath(Registry& reg, int32_t context, int64_t lane,
                        const std::vector<int64_t>& path) {
    const auto site_of = [&reg](int64_t id) -> Site* {
        if (id <= 0) return nullptr;
        return FindSite(reg, id);
    };
    if (path.empty()) return nullptr;
    Site* first = site_of(path[0]);
    if (first == nullptr) return nullptr;
    auto it = first->slot_of_key.find(SiteKey{context, lane});
    if (it == first->slot_of_key.end()) return nullptr;
    Slot* current = FindSlot(reg, it->second);
    for (size_t k = 1; current != nullptr && k < path.size(); ++k) {
        Site* site = site_of(path[k]);
        if (site == nullptr) return nullptr;
        if (current->site == site->id) continue;
        const SiteKey key{context, current->id};
        auto forward = site->forward_of_key.find(key);
        if (forward != site->forward_of_key.end() && forward->second == current->id) continue;
        auto held = site->slot_of_key.find(key);
        if (held == site->slot_of_key.end()) return nullptr;
        current = FindSlot(reg, held->second);
    }
    if (current == nullptr || current->context != context) return nullptr;
    return current;
}

// Make every side of `slot` stand for no handle: the next solve on it factors from its values.
void InvalidateSlot(Slot& slot) {
    if (slot.primary) slot.primary->generation = 0;
    if (slot.batch_primary) {
        std::fill(slot.batch_primary->generations.begin(),
                  slot.batch_primary->generations.end(), 0);
    }
}

// The slots `lanes` read handle lanes name. A lane whose slot id is negative is empty (it holds no
// factor) and is marked in `empty`; `first` receives the first lane that is not empty, -1 when
// every lane is (`batch` and `singles` then stay empty). The other lanes name lanes of one batch
// slot, lane l its lane l, each at the generation of its own (`batch`; an empty lane stands for
// its lane), or single slots only (`singles`, one per lane, null for an empty lane).
ffi::Error ClassifyLanes(Registry& reg, int64_t lanes, const int32_t* contexts,
                         const int32_t* ids, Slot** batch,
                         std::vector<Slot*>* singles, std::vector<uint8_t>* empty,
                         int64_t* first) {
    *batch = nullptr;
    singles->clear();
    empty->assign(static_cast<size_t>(lanes), 0);
    *first = -1;
    for (int64_t l = 0; l < lanes; ++l) {
        if (ids[l] < 0) {
            (*empty)[l] = 1;
        } else if (*first < 0) {
            *first = l;
        }
    }
    if (*first < 0) return ffi::Error::Success();
    const int64_t f = *first;
    Slot* found = nullptr;
    int32_t lane = 0;
    CONTRACT_TRY(FindHeld(reg, contexts[f], ids[f], &found, &lane));
    if (found->batch) {
        bool ordered = lane == f && found->lanes == lanes;
        for (int64_t l = f + 1; ordered && l < lanes; ++l) {
            if ((*empty)[l]) continue;
            ordered = contexts[l] == contexts[f] &&
                      static_cast<int64_t>(ids[l]) == static_cast<int64_t>(found->id) + l;
        }
        if (!ordered) return LaneOrderError(lanes);
        *batch = found;
        return ffi::Error::Success();
    }
    singles->assign(static_cast<size_t>(lanes), nullptr);
    for (int64_t l = 0; l < lanes; ++l) {
        if ((*empty)[l]) continue;
        Slot* slot = nullptr;
        CONTRACT_TRY(FindHeld(reg, contexts[l], ids[l], &slot, &lane));
        if (slot->batch) return LaneOrderError(lanes);
        (*singles)[l] = slot;
    }
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// cuDSS work on one side
// --------------------------------------------------------------------------

// Upload the pattern, with its assembly map, to `device` unless it is there already.
ffi::Error EnsurePattern(Registry& reg, Analysis& a, int device, cudaStream_t stream) {
    if (a.On(device).offsets != nullptr) {
        return ffi::Error::Success();
    }
    const bool first = !a.DeviceResident();
    struct Upload {
        const void* host;
        size_t bytes;
    };
    const Upload uploads[5] = {
        {a.offsets.data(), static_cast<size_t>(a.n + 1) * sizeof(int32_t)},
        {a.columns.data(), static_cast<size_t>(a.nse) * sizeof(int32_t)},
        {a.gather_offsets.data(), static_cast<size_t>(a.nse + 1) * sizeof(int32_t)},
        {a.gather_entries.data(), static_cast<size_t>(a.nnz) * sizeof(int32_t)},
        {a.diagonal_add.data(), a.diagonal_add.size()},
    };
    void* targets[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < 5 && err == cudaSuccess; ++i) {
        if (uploads[i].bytes == 0) continue;
        err = cudaMalloc(&targets[i], uploads[i].bytes);
        if (err == cudaSuccess) {
            err = cudaMemcpyAsync(targets[i], uploads[i].host, uploads[i].bytes,
                                  cudaMemcpyHostToDevice, stream);
        }
    }
    if (err != cudaSuccess) {
        for (void* target : targets) {
            if (target != nullptr) cudaFree(target);
        }
        return CudaFailed(err, "pattern upload");
    }
    DevicePattern& pattern = a.patterns[device];
    pattern.offsets = static_cast<int32_t*>(targets[0]);
    pattern.columns = static_cast<int32_t*>(targets[1]);
    pattern.gather_offsets = static_cast<int32_t*>(targets[2]);
    pattern.gather_entries = static_cast<int32_t*>(targets[3]);
    pattern.diagonal_add = static_cast<uint8_t*>(targets[4]);
    if (first) ++reg.device_analyses;
    return ffi::Error::Success();
}

// The values a call carries: `nnz` entries per lane in the caller's coordinate order (one per
// coordinate of the topology, duplicates included), float64 when `f64` else float32, lane i at
// `data` + i * nnz entries. The handlers assemble them into CSR order on the device.
struct RawValues {
    const void* data = nullptr;
    bool f64 = true;
    int64_t nnz = 0;

    size_t Size() const { return f64 ? sizeof(double) : sizeof(float); }
    const void* Lane(int64_t lane) const {
        return static_cast<const char*>(data) +
               static_cast<size_t>(lane) * static_cast<size_t>(nnz) * Size();
    }
};

// Refuse values of another coordinate count than the analysed topology's.
ffi::Error CheckRaw(const Analysis& a, const RawValues& raw) {
    if (raw.nnz != a.nnz) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cuDSS values operand has " + std::to_string(raw.nnz) +
                              " entries per lane; the analysed topology has " +
                              std::to_string(a.nnz) +
                              " coordinate entries. Pass the values of the topology the factor "
                              "was analysed with, one per pattern entry.");
    }
    return ffi::Error::Success();
}

// Assemble lanes [first, first + count) of `raw` into `out`: the CSR values of `slot`'s dtype,
// lane `first` of the run at `out`, with one kernel over the analysis's device-resident assembly
// map (uploaded with the pattern). Nothing is waited for.
ffi::Error AssembleLanes(Registry& reg, Slot& slot, const RawValues& raw, int64_t first,
                         int64_t count, void* out, cudaStream_t stream) {
    if (count <= 0) return ffi::Error::Success();
    Analysis& a = *slot.analysis;
    CONTRACT_TRY(EnsurePattern(reg, a, slot.device, stream));
    using diff_soup_assembly::Scalar;
    const DevicePattern& pattern = a.On(slot.device);
    diff_soup_assembly::AssemblyMap map;
    map.offsets = pattern.gather_offsets;
    map.entries = pattern.gather_entries;
    map.diagonal_add = pattern.diagonal_add;
    map.nnz = a.nnz;
    map.nse = a.nse;
    CONTRACT_CUDA(diff_soup_assembly::QueueAssembly(
                      stream, map, raw.f64 ? Scalar::kFloat64 : Scalar::kFloat32,
                      raw.Lane(first), slot.dtype == CUDA_R_64F ? Scalar::kFloat64 : Scalar::kFloat32,
                      out, count),
                  "value assembly kernel");
    ++reg.stats.assemblies;
    reg.stats.assembled_lanes += count;
    return ffi::Error::Success();
}

// Where a side's value copy is filled from: the call's coordinate values from lane `lane` on
// (`raw`, assembled into the side's pattern order), CSR values already in that order (`csr`, a
// device pointer), or nothing (the copy holds the values already).
struct ValueFill {
    const RawValues* raw = nullptr;
    int64_t lane = 0;
    const void* csr = nullptr;

    static ValueFill Raw(const RawValues& values, int64_t lane = 0) {
        ValueFill fill;
        fill.raw = &values;
        fill.lane = lane;
        return fill;
    }
    static ValueFill Csr(const void* values) {
        ValueFill fill;
        fill.csr = values;
        return fill;
    }
};

// Fill `count` lanes of a value copy from `fill`, lane 0 of the copy at `target`. Nothing is
// waited for.
ffi::Error FillValues(Registry& reg, Slot& slot, const ValueFill& fill, void* target,
                      int64_t count, cudaStream_t stream) {
    if (fill.raw != nullptr) {
        return AssembleLanes(reg, slot, *fill.raw, fill.lane, count, target, stream);
    }
    if (fill.csr != nullptr) {
        const size_t bytes = static_cast<size_t>(count) *
                             static_cast<size_t>(slot.analysis->nse) * slot.value_size;
        CONTRACT_CUDA(cudaMemcpyAsync(target, fill.csr, bytes, cudaMemcpyDeviceToDevice, stream),
                      "cudaMemcpyAsync (factor values)");
    }
    return ffi::Error::Success();
}

ffi::Error GrowScratch(ScanScratch& scratch, size_t device_bytes, size_t host_bytes,
                       ScanScratch** out);

// The per-lane words of a value check (assembly_kernels.h: kLaneChanged, kLaneAsks, kLaneEmpty):
// two device words per lane on the slot's device (the fixed bits of lane i at word i, its
// changed bit at word lanes + i), read back with one copy and one wait and decoded into the
// first `lanes` host words, one word of bits per lane.
struct ValueCheck {
    ScanScratch* scratch = nullptr;
    int64_t lanes = 0;
    int32_t tag = 0;  // the words of this check carry it (QueueAssemblyCompare)

    int32_t* Device() const { return static_cast<int32_t*>(scratch->device_words); }
    const int32_t* Host() const { return static_cast<const int32_t*>(scratch->host_words); }
};

ffi::Error BeginValueCheck(Registry& reg, const Slot& slot, int64_t lanes, ValueCheck* check) {
    const size_t bytes = 2 * static_cast<size_t>(std::max<int64_t>(lanes, 1)) * sizeof(int32_t);
    ScanScratch& scratch = reg.flags[slot.device];
    const bool grows = scratch.device_bytes < bytes;
    CONTRACT_TRY(GrowScratch(scratch, bytes, bytes, &check->scratch));
    if (grows) {
        // Fresh words hold no tag of a check (tags are positive).
        CONTRACT_CUDA(cudaMemset(check->scratch->device_words, 0, check->scratch->device_bytes),
                      "cudaMemset (value check words)");
    }
    reg.check_tag = reg.check_tag >= (1 << 28) - 1 ? 1 : reg.check_tag + 1;
    check->tag = reg.check_tag;
    check->lanes = lanes;
    return ffi::Error::Success();
}

// Queue the assembly of lanes [first, first + count) of `raw` into `out` (the value copy of a
// side of `slot`, lane `first` at `out`), in place, comparing each
// entry with the one it replaces, with the lanes' words from word `word` of `check` on. With
// `predicate` (device, one byte per lane of the call), a lane whose predicate is false assembles
// `held` instead and a lane whose `held_slot` entry (device) is negative is left alone; see
// QueueAssemblyCompare, which also writes each lane's source values into `kept` (the caller's
// layout, lane `first` at `kept`) and the `words` of a one-lane call when they are given.
// Nothing is waited for.
ffi::Error QueueValueCheck(Registry& reg, Slot& slot, const RawValues& raw,
                           const RawValues* held, const uint8_t* predicate,
                           const int32_t* held_slot, int64_t first, int64_t count, void* out,
                           const ValueCheck& check, int64_t word, cudaStream_t stream,
                           void* kept = nullptr,
                           const diff_soup_assembly::ResultWords* words = nullptr) {
    if (count <= 0) return ffi::Error::Success();
    Analysis& a = *slot.analysis;
    CONTRACT_TRY(EnsurePattern(reg, a, slot.device, stream));
    using diff_soup_assembly::Scalar;
    const DevicePattern& pattern = a.On(slot.device);
    diff_soup_assembly::AssemblyMap map;
    map.offsets = pattern.gather_offsets;
    map.entries = pattern.gather_entries;
    map.diagonal_add = pattern.diagonal_add;
    map.nnz = a.nnz;
    map.nse = a.nse;
    CONTRACT_CUDA(diff_soup_assembly::QueueAssemblyCompare(
                      stream, map, raw.f64 ? Scalar::kFloat64 : Scalar::kFloat32, raw.Lane(first),
                      held != nullptr ? held->Lane(first) : nullptr,
                      predicate != nullptr ? predicate + first : nullptr,
                      held_slot != nullptr ? held_slot + first : nullptr,
                      slot.dtype == CUDA_R_64F ? Scalar::kFloat64 : Scalar::kFloat32, out, count,
                      check.Device() + word, check.Device() + check.lanes + word, check.tag, kept,
                      words),
                  "value check kernel");
    ++reg.stats.assemblies;
    ++reg.stats.value_checks;
    reg.stats.assembled_lanes += count;
    return ffi::Error::Success();
}

// Queue the copy of the words of every lane of `check` back to its host words (no wait).
ffi::Error QueueValueCheckRead(const ValueCheck& check, cudaStream_t stream) {
    const size_t bytes = 2 * static_cast<size_t>(check.lanes) * sizeof(int32_t);
    CONTRACT_CUDA(cudaMemcpyAsync(check.scratch->host_words, check.scratch->device_words, bytes,
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync (value check words)");
    return ffi::Error::Success();
}

// Decode the host words of `check`, once its copy has completed: a word carrying the check's tag
// gives its bits, any other word (a lane this check did not mark) gives 0.
void DecodeValueCheck(const ValueCheck& check) {
    auto* words = static_cast<int32_t*>(check.scratch->host_words);
    const auto bits = [&](int32_t word) { return (word >> 3) == check.tag ? (word & 7) : 0; };
    for (int64_t l = 0; l < check.lanes; ++l) {
        words[l] = bits(words[l]) | bits(words[check.lanes + l]);
    }
}

// Read the words of every lane of `check` back to its host words (one copy and one wait) and
// decode them (DecodeValueCheck).
ffi::Error ReadValueCheck(const ValueCheck& check, cudaStream_t stream) {
    CONTRACT_TRY(QueueValueCheckRead(check, stream));
    CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (value check words)");
    DecodeValueCheck(check);
    return ffi::Error::Success();
}

// The precision options of the analysis on a configuration of `slot`, for float64 factors only
// (both act on float64 factors, and a float32 factor has nothing to downcast): the downcast of
// the factors to float32 storage (CUDSS_CONFIG_DOWNCAST_FACTORS), and the emulated float64
// factorization with the given mantissa bits (CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT).
ffi::Error ConfigurePrecision(const Analysis& a, const Slot& slot, cudssConfig_t config) {
    if (slot.dtype != CUDA_R_64F) return ffi::Error::Success();
    if (Downcast(a, slot)) {
        cudssDowncastFactorsMode_t mode = static_cast<cudssDowncastFactorsMode_t>(DowncastMode(a, slot));
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_DOWNCAST_FACTORS, &mode, sizeof(mode)),
                       "cudssConfigSet downcast_factors of a held factor");
    }
    if (a.emulation_mantissa_bits > 0) {
        int bits = a.emulation_mantissa_bits;
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_EMULATION_MANTISSA_BIT_COUNT, &bits,
                                      sizeof(bits)),
                       "cudssConfigSet emulation_mantissa_bits of a held factor");
    }
    return ffi::Error::Success();
}

// The options of the analysis on a new configuration of `slot`: pivot perturbation off,
// matching with scaling except on symmetric positive definite sides (matching off there),
// reordering, iterative refinement, hybrid memory, deterministic mode, the diagnostics the
// environment asks for, and the slot's model group (ConfigureModelGroup). The matrix type (Slot::MatrixType) decides whether the matching
// preserves symmetry: only on a side factored with cuDSS's symmetric type.
// The hybrid device memory limit is set after the analysis (AnalyzeSide).
ffi::Error Configure(const Analysis& a, const Slot& slot, cudssConfig_t config) {
    CONTRACT_TRY(ApplyDiagnosticConfig(slot.handle, config));
    // CUDSS_CONFIG_PIVOT_EPSILON is a double for both precisions.
    const double epsilon = 0.0;
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON, &epsilon, sizeof(epsilon)),
                   "cudssConfigSet pivot_epsilon of a held factor");
    int matching = slot.Matched() ? CUDSS_MATCHING_ALG_AUTO : CUDSS_MATCHING_ALG_NONE;
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof(matching)),
                   "cudssConfigSet matching of a held factor");
    if (a.reorder_alg >= 0) {
        int alg = a.reorder_alg;
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)),
                       "cudssConfigSet reordering of a held factor");
    }
    int ir = RefinementSteps(a, slot);
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                   "cudssConfigSet ir_nsteps of a held factor");
    CONTRACT_TRY(ConfigurePrecision(a, slot, config));
    if (a.hybrid_mode != 0) {
        int on = 1;
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &on, sizeof(on)),
                       "cudssConfigSet hybrid_mode of a held factor");
    }
    if (a.deterministic != 0) {
        int on = 1;
        CONTRACT_CUDSS(
            cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE, &on, sizeof(on)),
            "cudssConfigSet deterministic of a held factor");
    }
    return ConfigureModelGroup(slot, config);
}

double Seconds(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Under hybrid memory, after the analysis of `data`: record cuDSS's device memory floor for the
// analysed factorization (CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN, reported by
// contract_analysis_info), refuse a hybrid_device_memory_limit_mb below it with both numbers,
// and set the limit on `config`. cuDSS includes the solve's workspace in the floor only once the
// analysis has run, and rejects a smaller limit at factorization, so the limit is set here and
// never before the analysis. An unreadable floor leaves the check to cuDSS.
ffi::Error ApplyHybridLimit(Analysis& a, const Slot& slot, cudssData_t data,
                            cudssConfig_t config) {
    if (a.hybrid_mode == 0) return ffi::Error::Success();
    int64_t floor_bytes = -1;
    size_t written = 0;
    const bool known = cudssDataGet(slot.handle, data, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                                    &floor_bytes, sizeof(floor_bytes),
                                    &written) == CUDSS_STATUS_SUCCESS;
    if (known) a.hybrid_device_memory_min = floor_bytes;
    if (a.hybrid_device_memory_limit_mb <= 0) return ffi::Error::Success();
    if (known) {
        const std::string refusal = HybridLimitRefusal(floor_bytes, a.hybrid_device_memory_limit_mb);
        if (!refusal.empty()) return ffi::Error(ffi::ErrorCode::kInvalidArgument, refusal);
    }
    const int64_t limit = a.hybrid_device_memory_limit_mb * 1024 * 1024;
    return SetHybridDeviceLimit(config, limit, slot.mg_indices, slot.device);
}

// Pass the seed of sides without matching to `data` before its analysis: the kept permutation
// and, when one is kept, the partition tree of the same analysis. Returns cuDSS's status for the
// permutation; `with_tree` receives whether cuDSS took the tree too (a tree it refuses leaves the
// seed as the permutation alone).
cudssStatus_t SetSeed(const Analysis& a, const Slot& slot, cudssData_t data, bool* with_tree) {
    *with_tree = false;
    const cudssStatus_t status =
        cudssDataSet(slot.handle, data, CUDSS_DATA_USER_PERM, a.tree_permutation.data(),
                     static_cast<size_t>(a.n) * sizeof(int32_t));
    if (status == CUDSS_STATUS_SUCCESS && !a.nd_tree.empty()) {
        *with_tree = cudssDataSet(slot.handle, data, CUDSS_DATA_USER_ND_PARTITION_TREE,
                                  a.nd_tree.data(), a.nd_tree.size()) == CUDSS_STATUS_SUCCESS;
    }
    return status;
}

// After a cold analysis of `data`: keep the permutation of the record's first cold analysis
// (reported, never passed to a matched side), and, from the first cold analysis of a side without
// matching (`matched` false), its permutation and partition tree as that kind of side's seed.
void KeepColdOrdering(Analysis& a, const Slot& slot, cudssData_t data, bool matched) {
    const bool keep_permutation = static_cast<int32_t>(a.permutation.size()) != a.n;
    const bool keep_tree_seed = !matched && static_cast<int32_t>(a.tree_permutation.size()) != a.n;
    if (!keep_permutation && !keep_tree_seed) return;
    std::vector<int32_t> permutation(static_cast<size_t>(a.n), -1);
    size_t written = 0;
    const cudssStatus_t status =
        cudssDataGet(slot.handle, data, CUDSS_DATA_PERM_REORDER_ROW, permutation.data(),
                     permutation.size() * sizeof(int32_t), &written);
    if (status != CUDSS_STATUS_SUCCESS || written != permutation.size() * sizeof(int32_t)) return;
    if (keep_tree_seed) {
        // The partition tree of the same analysis, asked for its size first.
        std::vector<char> tree;
        size_t tree_bytes = 0;
        if (cudssDataGet(slot.handle, data, CUDSS_DATA_ND_PARTITION_TREE, nullptr, 0,
                         &tree_bytes) == CUDSS_STATUS_SUCCESS &&
            tree_bytes > 0) {
            tree.resize(tree_bytes);
            size_t tree_written = 0;
            if (cudssDataGet(slot.handle, data, CUDSS_DATA_ND_PARTITION_TREE, tree.data(),
                             tree.size(), &tree_written) != CUDSS_STATUS_SUCCESS ||
                tree_written != tree_bytes) {
                tree.clear();
            }
        }
        a.tree_unavailable = tree.empty();
        a.nd_tree = std::move(tree);
        a.tree_permutation = permutation;
    }
    if (keep_permutation) a.permutation = std::move(permutation);
}

// Run the analysis phase on a new side, waited for so its wall time can be reported. A side
// with matching always runs cold. With seeding on, a side without matching is seeded with the
// permutation and the partition tree of the first cold analysis of a side without matching,
// and runs cold (keeping them) while none is kept. A refused seed falls back to a cold
// analysis.
//
// The partition tree gives cuDSS the parallel schedule of the ordering; with the permutation
// alone it factors and solves on a slower schedule (same fill; measured 3x and 1.9x slower at
// n = 90,000, 1.09x and 1.8x at n = 1e6). The tree is kept from, and passed to, sides without
// matching only, always together with the permutation of the same analysis: matching makes
// the analysis depend on the values, and a tree of other values' matching fails or hangs
// cuDSS, while a permutation kept from other values can give a far worse fill (measured 150x
// on a matrix whose matching moved every row). A tree cuDSS refuses, or none reported, leaves
// the seed as the permutation alone.
ffi::Error AnalyzeSide(Registry& reg, Slot& slot, Side& side, cudaStream_t stream) {
    Analysis& a = *slot.analysis;
    const bool matched = slot.Matched();
    const std::vector<int32_t>& seed = a.tree_permutation;
    bool seeded = false;
    if (reg.seeding && !matched && !slot.OnModelGroup() &&
        static_cast<int32_t>(seed.size()) == a.n) {
        bool with_tree = false;
        cudssStatus_t status = SetSeed(a, slot, side.data, &with_tree);
        double seconds = 0.0;
        if (status == CUDSS_STATUS_SUCCESS) {
            const auto start = std::chrono::steady_clock::now();
            CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, side.config,
                                         side.data, side.A, side.x_scratch, side.b_scratch,
                                         &status));
            if (status == CUDSS_STATUS_SUCCESS) {
                CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
                seconds = Seconds(start);
            }
        }
        if (status == CUDSS_STATUS_SUCCESS) {
            seeded = true;
            ++a.seeded_analyses;
            ++reg.stats.seeded_analyses;
            if (with_tree) ++a.tree_seeded_analyses;
            a.analysis_seconds_seeded_last = seconds;
        } else {
            // cuDSS refused the seed: run a cold analysis on a fresh data object.
            ++a.seed_refusals;
            cudssDataDestroy(slot.handle, side.data);
            side.data = nullptr;
            CONTRACT_CUDSS(cudssDataCreate(slot.handle, &side.data),
                           "cudssDataCreate of a held factor");
        }
    }
    if (!seeded) {
        const auto start = std::chrono::steady_clock::now();
        cudssStatus_t analysed = CUDSS_STATUS_SUCCESS;
        CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, side.config, side.data,
                                     side.A, side.x_scratch, side.b_scratch, &analysed));
        CONTRACT_CUDSS(analysed, "cudssExecute analysis of a held factor");
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
        a.analysis_seconds_cold = Seconds(start);
        ++a.cold_analyses;
        ++reg.stats.cold_analyses;
        KeepColdOrdering(a, slot, side.data, matched);
    }
    side.seeded = seeded;
    side.analysis_current = true;
    int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
    QueryMemoryEstimates(slot.handle, side.data, estimates);
    side.permanent_device_bytes = std::max<int64_t>(estimates[0], 0);
    return ApplyHybridLimit(a, slot, side.data, side.config);
}

// A new single side of `slot` over the analysis's pattern, analysed with the values of `fill`,
// which also fill its value copy.
ffi::Error CreateSide(Registry& reg, Slot& slot, const ValueFill& fill, cudaStream_t stream,
                      std::unique_ptr<Side>* out) {
    Analysis& a = *slot.analysis;
    CONTRACT_TRY(EnsurePattern(reg, a, slot.device, stream));

    auto side = std::make_unique<Side>();
    side->handle = slot.handle;
    side->device = slot.device;
    CONTRACT_CUDSS(cudssConfigCreate(&side->config), "cudssConfigCreate of a held factor");
    CONTRACT_CUDSS(cudssDataCreate(slot.handle, &side->data), "cudssDataCreate of a held factor");
    CONTRACT_TRY(Configure(a, slot, side->config));
    if (Downcast(a, slot)) ++reg.stats.downcast_configurations;

    side->value_bytes = a.nse * static_cast<int64_t>(slot.value_size);
    side->scratch_bytes = 2 * static_cast<int64_t>(a.n) * static_cast<int64_t>(slot.value_size);
    CONTRACT_CUDA(cudaMalloc(&side->values, static_cast<size_t>(side->value_bytes)),
                  "cudaMalloc (factor value copy)");
    CONTRACT_CUDA(cudaMalloc(&side->scratch_b, static_cast<size_t>(a.n) * slot.value_size),
                  "cudaMalloc (factor scratch)");
    CONTRACT_CUDA(cudaMalloc(&side->scratch_x, static_cast<size_t>(a.n) * slot.value_size),
                  "cudaMalloc (factor scratch)");
    CONTRACT_CUDA(cudaMemsetAsync(side->scratch_b, 0, static_cast<size_t>(a.n) * slot.value_size,
                                  stream),
                  "cudaMemsetAsync (factor scratch)");
    CONTRACT_CUDA(cudaMemsetAsync(side->scratch_x, 0, static_cast<size_t>(a.n) * slot.value_size,
                                  stream),
                  "cudaMemsetAsync (factor scratch)");
    CONTRACT_CUDA(cudaMalloc(reinterpret_cast<void**>(&side->verdict),
                             sizeof(diff_soup_status::SideVerdictWords)),
                  "cudaMalloc (factor status words)");
    CONTRACT_CUDA(cudaMemsetAsync(side->verdict, 0, sizeof(diff_soup_status::SideVerdictWords),
                                  stream),
                  "cudaMemsetAsync (factor status words)");
    CONTRACT_TRY(FillValues(reg, slot, fill, side->values, 1, stream));
    const DevicePattern& pattern = a.On(slot.device);
    CONTRACT_CUDSS(cudssMatrixCreateCsr(&side->A, a.n, a.n, a.nse, pattern.offsets,
                                        nullptr, pattern.columns, side->values,
                                        CUDA_R_32I, CUDA_R_32I, slot.dtype, slot.MatrixType(),
                                        CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                   "cudssMatrixCreateCsr of a held factor");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&side->b_scratch, a.n, 1, a.n, side->scratch_b,
                                       slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a held factor's scratch vector");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&side->x_scratch, a.n, 1, a.n, side->scratch_x,
                                       slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a held factor's scratch vector");
    CONTRACT_TRY(AnalyzeSide(reg, slot, *side, stream));
    *out = std::move(side);
    return ffi::Error::Success();
}

// The side of single slot `slot`, created and analysed on first use with the values of `fill`,
// which fill the new side's value copy. It factors A_bc whatever the slot's orientation: an "AT"
// factor solves with it in transpose mode (SolveSingle).
ffi::Error EnsureSide(Registry& reg, Slot& slot, const ValueFill& fill, cudaStream_t stream,
                      Side** out) {
    std::unique_ptr<Side>& holder = slot.primary;
    if (holder) {
        *out = holder.get();
        return ffi::Error::Success();
    }
    CONTRACT_TRY(CreateSide(reg, slot, fill, stream, &holder));
    *out = holder.get();
    return ffi::Error::Success();
}

// The status scan scratch of `device`, with room for the given device and pinned host bytes.
ffi::Error GrowScratch(ScanScratch& scratch, size_t device_bytes, size_t host_bytes,
                       ScanScratch** out) {
    if (scratch.device_bytes < device_bytes) {
        if (scratch.device_words != nullptr) cudaFree(scratch.device_words);
        scratch.device_words = nullptr;
        scratch.device_bytes = 0;
        CONTRACT_CUDA(cudaMalloc(&scratch.device_words, device_bytes),
                      "cudaMalloc (status scan scratch)");
        scratch.device_bytes = device_bytes;
    }
    if (scratch.host_bytes < host_bytes) {
        if (scratch.host_words != nullptr) cudaFreeHost(scratch.host_words);
        scratch.host_words = nullptr;
        scratch.host_bytes = 0;
        CONTRACT_CUDA(cudaMallocHost(&scratch.host_words, host_bytes),
                      "cudaMallocHost (status scan scratch)");
        scratch.host_bytes = host_bytes;
    }
    *out = &scratch;
    return ffi::Error::Success();
}

ffi::Error ScanScratchFor(Registry& reg, int device, size_t device_bytes, size_t host_bytes,
                          ScanScratch** out) {
    return GrowScratch(reg.scan[device], device_bytes, host_bytes, out);
}

// What a status scan found in one lane: whether its values are finite, whether its factor
// diagonal is finite and, when it is, the smallest and largest magnitude of its entries.
struct LaneScan {
    bool values_finite = true;
    bool finite = true;     // the diagonal
    double smallest = 0.0;  // smallest diagonal magnitude
    double largest = 0.0;   // largest diagonal magnitude
    // A symmetric indefinite single side's status solve (SymmetricStatusSolve): whether it ran,
    // whether its error x - x* is finite and, when it is, its largest magnitude.
    bool probed = false;
    bool probe_finite = true;
    double probe_error = 0.0;
};

// Scan `lanes` lanes in one pass on the device (status_kernels.cu) and read the per-lane
// results back with one copy and one wait: `value_count` values per lane at `values` and
// `diagonal_count` factor diagonal entries per lane at `diagonal` (device, `dtype`, lane i at
// i * count, consecutive lanes; either may be null to skip it).
ffi::Error ScanLanes(Registry& reg, int device, cudaStream_t stream, cudaDataType_t dtype,
                     const void* values, int64_t value_count, const void* diagonal,
                     int64_t diagonal_count, int64_t lanes, std::vector<LaneScan>* out) {
    out->assign(static_cast<size_t>(lanes), LaneScan{});
    if (lanes <= 0) return ffi::Error::Success();
    using diff_soup_status::LaneStatusWords;
    const size_t bytes = static_cast<size_t>(lanes) * sizeof(LaneStatusWords);
    ScanScratch* scratch = nullptr;
    CONTRACT_TRY(ScanScratchFor(reg, device, bytes, bytes, &scratch));
    auto* words = static_cast<LaneStatusWords*>(scratch->device_words);
    CONTRACT_CUDA(diff_soup_status::QueueLaneStatusScan(stream, dtype == CUDA_R_64F, values,
                                                        value_count, diagonal, diagonal_count,
                                                        lanes, words),
                  "status scan kernel");
    CONTRACT_CUDA(cudaMemcpyAsync(scratch->host_words, words, bytes, cudaMemcpyDeviceToHost,
                                  stream),
                  "cudaMemcpyAsync (status scan results)");
    CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (status scan results)");
    const auto* host = static_cast<const LaneStatusWords*>(scratch->host_words);
    const bool scanned_diagonal = diagonal != nullptr && diagonal_count > 0;
    for (int64_t l = 0; l < lanes; ++l) {
        LaneScan& scan = (*out)[static_cast<size_t>(l)];
        scan.values_finite = host[l].values_nonfinite == 0;
        scan.finite = host[l].diagonal_nonfinite == 0;
        if (scanned_diagonal && scan.finite) {
            scan.smallest = diff_soup_status::SmallestMagnitude(host[l]);
            scan.largest = diff_soup_status::LargestMagnitude(host[l]);
        }
    }
    return ffi::Error::Success();
}

// The unit roundoff of the factor's scalar type: 2^-53 for float64, 2^-24 for float32.
double UnitRoundoff(cudaDataType_t dtype) {
    return std::ldexp(1.0, dtype == CUDA_R_64F ? -53 : -24);
}

// A matched side's smallest pivot at or below this many unit roundoffs of its largest marks
// the factor singular. An exactly dependent column leaves a remainder of up to about two unit
// roundoffs on the GPU (measured), and every regular matrix measured sits above one tenth of
// its largest pivot after equilibration, so sixteen keeps an eightfold margin on both sides.
constexpr double kSingularPivotRoundoffs = 16.0;

// A batch lane whose smallest pivot is at most this ratio of its largest is factored again
// alone with matching: the square root of the unit roundoff. Batches run without matching, so
// a badly scaled regular lane shows a tiny pivot (a row scaled by 1e-20 gave a ratio of 1e-20
// and a wrong finite solution, measured) that a matched factor equilibrates away; above this
// ratio the batch factor serves the lane.
double BatchSuspectRatio(cudaDataType_t dtype) { return std::sqrt(UnitRoundoff(dtype)); }

// A symmetric indefinite factor whose status solve misses x* by more than this fraction of x*'s
// largest entry is singular. The error of a backward-stable solve grows like the condition
// number times the unit roundoff, so this bound sits where a general side's 16-roundoff pivot
// ratio does (a condition number near 1 / (16 u)). Measured with cuDSS 1.0.0.35 on an RTX
// 4070 Ti (symmetric factorization with matching, pivot epsilon 0, n = 400 to 900): singular
// saddle-point and 2 x 2 block matrices gave non-finite errors, an exactly singular Neumann
// Laplacian 0.86 times that largest entry, and regular 2 x 2 block matrices 4.2e-4 times it at a
// condition number near 1e13 and 7.1e-2 times it near 1e15.
constexpr double kSymmetricProbeError = 1.0 / 16.0;

// Read the factor's diagonal, which cuDSS reports after a numeric phase (CUDSS_DATA_DIAG), into
// the pinned host buffer `host` and scan it on the host into `scan`: whether every entry is
// finite and, when they are, their smallest and largest magnitude. The library waits inside
// this call for all the work queued on the handle's stream (also work queued after the numeric
// phase, measured), so every copy the caller queued before it has completed when it returns;
// a query of the stream confirms it.
ffi::Error ReadDiagonalHost(const Slot& slot, Side& side, cudaStream_t stream, void* host,
                            LaneScan* scan) {
    const int64_t n = slot.analysis->n;
    const size_t bytes = static_cast<size_t>(n) * slot.value_size;
    size_t written = 0;
    CONTRACT_TRY(HandleEnter(slot, stream));
    const cudssStatus_t read = cudssDataGet(slot.handle, side.data, CUDSS_DATA_DIAG, host, bytes,
                                            &written);
    CONTRACT_TRY(HandleLeave(slot, stream));
    CONTRACT_CUDSS(read, "cudssDataGet diag of a held factor");
    // The stream is idle by now (the library waited for it); a query confirms it without a
    // second wait, which runs only if the library ever returns earlier.
    const cudaError_t idle = cudaStreamQuery(stream);
    if (idle == cudaErrorNotReady) {
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (factor status)");
    } else if (idle != cudaSuccess) {
        return CudaFailed(idle, "cudaStreamQuery (factor status)");
    }
    if (written != bytes) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "cuDSS wrote " + std::to_string(written) +
                              " bytes for the diagonal of a held factor; its " +
                              std::to_string(n) + " entries take " + std::to_string(bytes) +
                              " bytes.");
    }
    *scan = LaneScan{};
    double smallest = std::numeric_limits<double>::infinity();
    double largest = 0.0;
    const bool f64 = slot.dtype == CUDA_R_64F;
    for (int64_t i = 0; i < n; ++i) {
        const double entry = f64 ? static_cast<const double*>(host)[i]
                                 : static_cast<double>(static_cast<const float*>(host)[i]);
        if (!std::isfinite(entry)) {
            scan->finite = false;
            return ffi::Error::Success();
        }
        const double magnitude = std::fabs(entry);
        smallest = std::min(smallest, magnitude);
        largest = std::max(largest, magnitude);
    }
    scan->smallest = smallest;
    scan->largest = largest;
    return ffi::Error::Success();
}

// Whether a factor whose diagonal scanned as `scan` is singular. It is singular when the
// diagonal holds a non-finite or a zero entry: with pivot perturbation off, an exactly zero
// pivot stays zero and the entries computed from it become infinite or nan. On a side that ran
// matching (any but a symmetric positive definite one), cuDSS has also equilibrated the matrix,
// and the factor is singular when its smallest pivot is at most 16 unit roundoffs of its
// largest: after equilibration a pivot that small means the matrix is singular to working
// precision. An exactly dependent column leaves a rounding remainder of about one unit roundoff
// instead of a zero pivot, and a scaled matrix with a pivot ratio that small has no correct
// digit in its solution anyway. Symmetric positive definite sides run without matching, so
// their diagonal is in the caller's units and only the zero and non-finite conditions apply.
// Single-precision factors also take only the zero and non-finite conditions: at float32 the
// pivot ratios of regular and of singular matrices overlap (a regular matrix of condition
// number 1.6e4 reaches 2.3e-9, below 16 float32 roundoffs, while an exactly dependent column
// leaves a remainder of a few float32 roundoffs), so no ratio separates them; whether a float32
// factor is accurate enough is decided by refinement against the float64 residual.
// `ratio` receives the smallest pivot magnitude over the largest (nan when non-finite).
//
// A side factored with cuDSS's symmetric indefinite factorization (Slot::SymmetricIndefinite) is
// judged by its status solve instead of by the zero and ratio conditions, which do not apply
// to a diagonal of D with 2 x 2 pivot blocks: it is singular when the diagonal holds a
// non-finite entry, or when the status solve's error is non-finite or, in float64, larger than
// kSymmetricProbeError times the largest entry of x*.
bool DiagonalSingular(const Slot& slot, const LaneScan& scan, double* ratio) {
    if (!scan.finite) {
        *ratio = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
    *ratio = scan.largest > 0.0 ? scan.smallest / scan.largest : 0.0;
    if (scan.probed) {
        return !scan.probe_finite ||
               (slot.dtype == CUDA_R_64F &&
                scan.probe_error >
                    kSymmetricProbeError * diff_soup_status::kProbeSolutionLargest);
    }
    return scan.smallest == 0.0 ||
           (slot.Matched() && slot.dtype == CUDA_R_64F &&
            scan.smallest <= kSingularPivotRoundoffs * UnitRoundoff(slot.dtype) * scan.largest);
}

// A failed numeric phase. Out of memory keeps the device report of CudssCallFailed; any other
// status is named, and under hybrid memory a device limit below cuDSS's floor (refused after the
// analysis whenever the floor is known) is reported with the limit that would satisfy it. A CUDA
// error pending on the thread is named too.
ffi::Error FactorizationFailed(const Analysis& a, cudssStatus_t status, bool refactor,
                               int64_t peak_device_estimate) {
    const char* phase = refactor ? "cudssExecute refactorization of a held factor"
                                 : "cudssExecute factorization of a held factor";
    if (status == CUDSS_STATUS_ALLOC_FAILED) {
        return CudssCallFailed(status, phase, peak_device_estimate);
    }
    const std::string refusal =
        a.hybrid_mode != 0
            ? HybridLimitRefusal(a.hybrid_device_memory_min, a.hybrid_device_memory_limit_mb)
            : std::string();
    const std::string cuda = PendingCudaError();
    if (!refusal.empty()) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          std::string("cuDSS ") + phase + " failed with status " +
                              std::to_string(static_cast<int>(status)) + ": " + refusal + cuda);
    }
    return ffi::Error(ffi::ErrorCode::kInternal, std::string("cuDSS ") + phase +
                                                     " failed with status " +
                                                     std::to_string(static_cast<int>(status)) +
                                                     "." + cuda);
}

// Run one numeric phase (factorization, or refactorization when `refactor`) on `side` over the
// values in its value copy and queue its verdict on the device: one kernel sets the side's
// device status word to NONFINITE when a value is nan or infinite and to SUCCESS otherwise
// (QueueValuesVerdict; a solve adds SINGULAR, QueueSolveVerdict). The host does not wait: the
// verdict stays pending until SettleSide reads it. `stale` says whether the phase is a full
// factorization on an analysis of other values. Nothing else is read from the library on this
// path: with pivot perturbation off no pivot is perturbed and a zero pivot shows in the
// diagonal, so CUDSS_DATA_INFO and CUDSS_DATA_NPIVOTS, each a host read of its own inside the
// library, are not consulted. The verdict kernel also writes `result` when it is given.
ffi::Error NumericPhase(Registry& reg, Slot& slot, Side& side, cudaStream_t stream,
                        bool refactor, bool stale,
                        const diff_soup_status::VerdictResult* result = nullptr) {
    Analysis& a = *slot.analysis;
    const cudssPhase_t phase = refactor ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
    CONTRACT_TRY(SetSolveMode(side, CUDSS_SOLVE_MODE_NORMAL));
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, phase, side.config, side.data, side.A,
                                 side.x_scratch, side.b_scratch, &status));
    if (status != CUDSS_STATUS_SUCCESS) {
        side.status = kSingular;
        side.factored = false;
        side.pending = false;
        side.retry = kRetryNone;
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, side.data, estimates);
        return FactorizationFailed(a, status, refactor, estimates[1]);
    }
    side.factored = true;
    CONTRACT_CUDA(diff_soup_status::QueueValuesVerdict(stream, slot.dtype == CUDA_R_64F,
                                                       side.values, a.nse, side.verdict, result),
                  "factor status kernel");
    side.pending = true;
    side.pending_refactorization = refactor;
    side.pending_stale = stale && !refactor;
    return ffi::Error::Success();
}

// Settle the pending verdict of `side` from its device status word (`word`, read back) and the
// scan of its factor's diagonal by the rules of "Status" above: NONFINITE when the word says the
// values were not finite (the next request then factors afresh), else SINGULAR by
// DiagonalSingular, else SUCCESS. A settled status other than the word is written back to the
// device word, so the solves queued from now on take it. A refactorization that ended singular
// asks the side's next numeric phase for a full factorization, and a full factorization on an
// analysis of other values that ended singular asks for a new analysis first (Side::retry);
// only a singular verdict that asks for nothing more is counted as singular. `deferred` says
// whether the verdict is read after the call that ran the phase (counted in deferred_status_reads).
ffi::Error ApplyVerdict(Registry& reg, Slot& slot, Side& side, int32_t word, bool deferred,
                        const LaneScan& scan, cudaStream_t stream) {
    Analysis& a = *slot.analysis;
    int32_t status = kSuccess;
    side.retry = kRetryNone;
    if (word == kNonfinite) {
        status = kNonfinite;
        side.factored = false;
        ++reg.stats.nonfinite;
    } else {
        double ratio = 0.0;
        if (DiagonalSingular(slot, scan, &ratio)) {
            status = kSingular;
            if (side.pending_refactorization) {
                side.retry = kRetryFresh;
            } else if (side.pending_stale) {
                side.retry = kRetryReanalyse;
            } else {
                ++reg.stats.singular;
            }
        }
        a.pivot_ratio = ratio;
    }
    side.status = status;
    side.pending = false;
    if (deferred) ++reg.stats.deferred_status_reads;
    if (status != word) {
        diff_soup_assembly::ResultWords words;
        words.count = 1;
        words.targets[0] = &side.verdict->status;
        words.values[0] = status;
        CONTRACT_CUDA(diff_soup_assembly::QueueResultWords(stream, words), "factor status write");
    }
    return ffi::Error::Success();
}

// The pinned host scratch of a settle on `device`: the status word in the first 16 bytes, then
// room for a diagonal of `n` entries of `value_size` bytes.
ffi::Error SettleScratch(Registry& reg, int device, int64_t n, size_t value_size,
                         ScanScratch** out) {
    return GrowScratch(reg.settle[device], 0, 16 + static_cast<size_t>(n) * value_size, out);
}

// The status solve of a symmetric indefinite single side, whose factor diagonal cannot decide
// its status (DiagonalSingular): solve A x = A x* for the known x* of the batch status solve
// (status_kernels.h) on the side's scratch vectors without iterative refinement (a downcast
// factor's with its solves' refinement, StatusRefinementSteps), and scan the
// error x - x* into the probe fields of `scan` (one more wait). Counted in status_solves.
ffi::Error SymmetricStatusSolve(Registry& reg, Slot& slot, Side& side, cudaStream_t stream,
                                LaneScan* scan) {
    Analysis& a = *slot.analysis;
    const DevicePattern& pattern = a.On(slot.device);
    const bool f64 = slot.dtype == CUDA_R_64F;
    CONTRACT_CUDA(diff_soup_status::QueueProbeRightHandSides(stream, f64, pattern.offsets,
                                                             pattern.columns, side.values, a.n,
                                                             a.nse, 1, side.scratch_b),
                  "status solve right-hand-side kernel");
    CONTRACT_TRY(SetSolveMode(side, CUDSS_SOLVE_MODE_NORMAL));
    const int ir_steps = RefinementSteps(a, slot);
    int ir = StatusRefinementSteps(a, slot);
    if (ir_steps != ir) {
        CONTRACT_CUDSS(cudssConfigSet(side.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                       "cudssConfigSet ir_nsteps of a held factor's status solve");
    }
    cudssStatus_t solved = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_SOLVE, side.config, side.data, side.A,
                                 side.x_scratch, side.b_scratch, &solved));
    if (ir_steps != ir) {
        ir = ir_steps;
        CONTRACT_CUDSS(cudssConfigSet(side.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                       "cudssConfigSet ir_nsteps of a held factor");
    }
    CONTRACT_CUDSS(solved, "cudssExecute status solve of a held factor");
    ++reg.stats.status_solves;
    CONTRACT_CUDA(diff_soup_status::QueueProbeErrors(stream, f64, side.scratch_x, a.n, 1),
                  "status solve error kernel");
    std::vector<LaneScan> errors;
    CONTRACT_TRY(ScanLanes(reg, slot.device, stream, slot.dtype, nullptr, 0, side.scratch_x, a.n,
                           1, &errors));
    scan->probed = true;
    scan->probe_finite = errors[0].finite;
    scan->probe_error = errors[0].finite ? errors[0].largest : 0.0;
    return ffi::Error::Success();
}

// Settle the pending verdict of `side` with one wait: the copy of its device status word is
// queued, then the diagonal read (ReadDiagonalHost) waits for both. Nothing happens when no
// verdict is pending. Work the caller queued before this call (a value check's copy) has
// completed when it returns, but only when a verdict was pending. `deferred` as ApplyVerdict.
// `read_word`, when given, receives the device status word as read (the settled status differs
// from it when the rules on the diagonal changed it).
ffi::Error SettleSide(Registry& reg, Slot& slot, Side& side, cudaStream_t stream,
                      bool deferred = true, int32_t* read_word = nullptr) {
    if (!side.pending) return ffi::Error::Success();
    ScanScratch* scratch = nullptr;
    CONTRACT_TRY(SettleScratch(reg, slot.device, slot.analysis->n, slot.value_size, &scratch));
    auto* word = static_cast<int32_t*>(scratch->host_words);
    *word = kSuccess;
    CONTRACT_CUDA(cudaMemcpyAsync(word, &side.verdict->status, sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync (factor status word)");
    LaneScan scan;
    CONTRACT_TRY(ReadDiagonalHost(slot, side, stream, static_cast<char*>(scratch->host_words) + 16,
                                  &scan));
    if (read_word != nullptr) *read_word = *word;
    if (slot.SymmetricIndefinite() && *word == kSuccess && scan.finite) {
        CONTRACT_TRY(SymmetricStatusSolve(reg, slot, side, stream, &scan));
    }
    return ApplyVerdict(reg, slot, side, *word, deferred, scan, stream);
}

// The status a call reports for `side` without waiting: the settled status, or, while a verdict
// is pending, `source` set to the side's device status word (the result is written from it).
int32_t SideStatus(const Side& side, const int32_t** source) {
    if (side.pending) {
        *source = &side.verdict->status;
        return kSuccess;
    }
    return side.status;
}

// Run the analysis phase of `side` again on a fresh data object, over the values now in its
// value copy (cold on a side with matching; seeded as AnalyzeSide says otherwise).
ffi::Error Reanalyze(Registry& reg, Slot& slot, Side& side, cudaStream_t stream) {
    // Once its cleanup has begun a data object is never used or destroyed again, even when the
    // destroy reports a failure, so the pointer is cleared before the status is checked.
    const cudssStatus_t destroyed = cudssDataDestroy(slot.handle, side.data);
    side.data = nullptr;
    side.factored = false;
    CONTRACT_CUDSS(destroyed, "cudssDataDestroy of a held factor");
    CONTRACT_CUDSS(cudssDataCreate(slot.handle, &side.data), "cudssDataCreate of a held factor");
    CONTRACT_TRY(SetSolveMode(side, CUDSS_SOLVE_MODE_NORMAL));
    CONTRACT_TRY(AnalyzeSide(reg, slot, side, stream));
    side.facts_known = false;
    ++reg.stats.reanalyses;
    return ffi::Error::Success();
}

// Factor the values of `fill` into `side`. `refactorization` asks for cuDSS's refactorization
// phase, which is used only when the side holds a factorization already and its last settled
// verdict asked for no full factorization (Side::retry). `ran` reports whether a numeric phase
// ran (on finite values, when its verdict was settled here); `used_refactorization` whether
// that phase was the refactorization. A verdict still pending on the side is settled first: it
// decides how this request factors (one wait, on work queued before the request). A full
// factorization then settles its own verdict at once (one wait); a refactorization does not wait
// for the device, and its verdict stays pending (see "Status"), unless `settle` asks for it.
// `result`, when given, is written by the verdict kernel of the request's numeric phase (not by
// the phase of a new analysis that a singular full factorization asks for); it stands when the
// side's verdict is still pending on return.
//
// cuDSS computes the matching permutation and scaling in the analysis phase, from the values
// present then. A side analysed with other values than the current ones can therefore meet an
// exactly zero pivot on a regular matrix whose large entries sit elsewhere (with pivot
// perturbation off). So when a full factorization of such a side ends singular, the side is
// analysed again with the current values and factored once more before its status stands; a
// matrix that is singular pays that one extra analysis. A side whose settled verdict asked for a
// new analysis is analysed again before it is factored.
ffi::Error FactorSide(Registry& reg, Slot& slot, Side& side, const ValueFill& fill,
                      cudaStream_t stream, bool refactorization, bool* ran,
                      bool* used_refactorization, bool settle = false,
                      const diff_soup_status::VerdictResult* result = nullptr) {
    *ran = false;
    *used_refactorization = false;
    CONTRACT_TRY(SettleSide(reg, slot, side, stream));
    const int retry = side.retry;
    side.retry = kRetryNone;
    // Until a verdict settles the status, the host holds no SUCCESS for the side: an error below
    // leaves it SINGULAR, never a SUCCESS of an earlier request.
    side.status = kSingular;
    Analysis& a = *slot.analysis;
    // The side's analysis saw these very values only on its first factorization.
    bool stale_analysis = !side.analysis_current;
    side.analysis_current = false;
    CONTRACT_TRY(FillValues(reg, slot, fill, side.values, 1, stream));
    if (retry == kRetryReanalyse) {
        CONTRACT_TRY(Reanalyze(reg, slot, side, stream));
        side.analysis_current = false;
        stale_analysis = false;
    }
    const bool refactor = refactorization && side.factored && retry == kRetryNone;
    CONTRACT_TRY(NumericPhase(reg, slot, side, stream, refactor, stale_analysis, result));
    *ran = true;
    *used_refactorization = refactor;
    // A phase on a model group has completed on return (ExecuteOnModelGroup), so its verdict is
    // settled at once.
    if (settle || !refactor || slot.OnModelGroup()) {
        CONTRACT_TRY(SettleSide(reg, slot, side, stream, false));
        if (side.retry == kRetryReanalyse) {
            side.retry = kRetryNone;
            CONTRACT_TRY(Reanalyze(reg, slot, side, stream));
            side.analysis_current = false;
            CONTRACT_TRY(NumericPhase(reg, slot, side, stream, false, false));
            CONTRACT_TRY(SettleSide(reg, slot, side, stream, false));
        }
        // A phase on non-finite values does not count as a factorization that ran.
        *ran = side.status != kNonfinite;
    }
    if (!side.facts_known) {
        // The size facts of the factor, read from the library once per side and analysis (each
        // a host read of its own inside the library): the permanent device bytes, for the
        // resident-memory report, and the fill of L + U, for the analysis report.
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, side.data, estimates);
        if (estimates[0] >= 0) side.permanent_device_bytes = estimates[0];
        const FactorFacts facts = QueryFactorFacts(slot.handle, side.data);
        if (facts.lu_nnz >= 0) a.lu_nnz = facts.lu_nnz;
        side.facts_known = true;
    }
    return ffi::Error::Success();
}

// Solve with `side` into `x` for the `total` right-hand-side entries at `b` (n contiguous per
// right-hand side), with the transpose of the side's matrix when `transpose` (cuDSS's transpose
// solve mode, counted in transposed_solves). The dense matrices are re-pointed at the call's
// buffers when they point at other ones, or recreated when the number of right-hand sides
// changed; the solution buffer is zeroed first. With `words` (a one-lane solve whose status the
// host knows), the kernel that zeroes the solution also writes those words. cuDSS's
// deterministic mode solves one right-hand side per call (a solve of more columns returns
// CUDSS_STATUS_NOT_SUPPORTED, measured with cuDSS 1.0.0.35 whatever the width the side was
// analysed with), so a side of a deterministic analysis solves several right-hand sides one
// column at a time.
ffi::Error SolveSide(Registry& reg, Slot& slot, Side& side, const void* b, void* x,
                     int64_t total, cudaStream_t stream, bool transpose,
                     const diff_soup_assembly::ResultWords* words = nullptr) {
    const int64_t n = slot.analysis->n;
    const int64_t nrhs = total / n;
    const bool by_column = slot.analysis->deterministic != 0 && nrhs > 1;
    const int64_t width = by_column ? 1 : nrhs;
    if (side.b == nullptr || side.x == nullptr || side.nrhs != width) {
        // The pointers are cleared before the status is checked, so a failed destroy is never
        // repeated by the destructor.
        if (side.b != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(side.b);
            side.b = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy b of a held factor");
        }
        if (side.x != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(side.x);
            side.x = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy x of a held factor");
        }
        CONTRACT_CUDSS(cudssMatrixCreateDn(&side.b, n, width, n, const_cast<void*>(b),
                                           slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
                       "cudssMatrixCreateDn b of a held factor");
        CONTRACT_CUDSS(
            cudssMatrixCreateDn(&side.x, n, width, n, x, slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
            "cudssMatrixCreateDn x of a held factor");
        side.nrhs = width;
        side.bound_b = b;
        side.bound_x = x;
    }
    const bool zero_fill = ZeroFills(*slot.analysis, slot);
    const size_t bytes = zero_fill ? static_cast<size_t>(total) * slot.value_size : 0;
    if (words != nullptr) {
        CONTRACT_CUDA(diff_soup_assembly::QueueFillAndWords(stream, x, bytes, 0, *words),
                      "solution zero fill and result words kernel");
    } else if (zero_fill) {
        CONTRACT_TRY(ZeroSolveOutput(x, bytes, stream, false));
    }
    if (zero_fill) ++reg.stats.zero_fills;
    CONTRACT_TRY(SetSolveMode(side, transpose ? CUDSS_SOLVE_MODE_TRANSPOSE
                                              : CUDSS_SOLVE_MODE_NORMAL));
    const size_t column_bytes = static_cast<size_t>(n) * slot.value_size;
    for (int64_t column = 0; column < (by_column ? nrhs : 1); ++column) {
        const void* bc = static_cast<const char*>(b) + column * column_bytes;
        void* xc = static_cast<char*>(x) + column * column_bytes;
        if (side.bound_b != bc) {
            side.bound_b = nullptr;
            CONTRACT_CUDSS(cudssMatrixSetValues(side.b, const_cast<void*>(bc)),
                           "cudssMatrixSetValues b of a held factor");
            side.bound_b = bc;
        }
        if (side.bound_x != xc) {
            side.bound_x = nullptr;
            CONTRACT_CUDSS(cudssMatrixSetValues(side.x, xc),
                           "cudssMatrixSetValues x of a held factor");
            side.bound_x = xc;
        }
        cudssStatus_t solved = CUDSS_STATUS_SUCCESS;
        CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_SOLVE, side.config, side.data,
                                     side.A, side.x, side.b, &solved));
        CONTRACT_CUDSS(solved, transpose ? "cudssExecute transposed solve of a held factor"
                                         : "cudssExecute solve of a held factor");
    }
    if (transpose) ++reg.stats.transposed_solves;
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// cuDSS work on a batch side
// --------------------------------------------------------------------------

// The most factor entries one batch tile holds over all its lanes (the nnz of L + U of a lane
// times the tile's lanes). A float32 uniform batch of tridiagonal 64-row lanes with
// 2,147,483,740 matrix entries in all (just above 2**31 - 1; a tridiagonal factor has no fill
// beyond its pattern) factored and solved to 1e-7 with cuDSS 1.0.0.35 on an NVIDIA L40S, so cuDSS
// does not refuse the aggregate at the int32 limit; larger batches were not tried. A tile stays
// at half that limit, which also leaves room for the workspace cuDSS sizes from the entry count.
constexpr int64_t kBatchFactorEntries = int64_t{1} << 30;

// The nnz of L + U per lane that sizes the tiles of a batch: the fill cuDSS reported for a batch
// tile of the analysis, or, before any batch tile was analysed, the number of pattern entries.
int64_t LaneFactorEntries(const Analysis& a) {
    return a.lu_nnz_batch > 0 ? a.lu_nnz_batch : std::max<int64_t>(a.nse, 1);
}

// Lanes per batch tile: as many as keep a tile's row and value counts within int32 and its factor
// entries (`lane_entries` per lane) within kBatchFactorEntries, or fewer when
// contract_set_batch_tile_lanes asks for fewer.
int64_t TileLanes(const Registry& reg, const Analysis& a, int64_t lane_entries) {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    int64_t lanes = std::min({limit / std::max<int64_t>(a.n, 1),
                              limit / std::max<int64_t>(a.nse, 1),
                              kBatchFactorEntries / std::max<int64_t>(lane_entries, 1)});
    lanes = std::max<int64_t>(lanes, 1);
    if (reg.batch_tile_lanes > 0) lanes = std::min(lanes, reg.batch_tile_lanes);
    return lanes;
}

// Whether a batch of `a` runs lane by lane, each lane on a single side of its own: cuDSS's uniform
// batch refuses deterministic mode and hybrid memory mode (CUDSS_STATUS_NOT_SUPPORTED at the
// analysis, measured with cuDSS 1.0.0.35). Batch tiles never run matching, whose explicit
// algorithms the uniform batch refuses too, so matching sends no batch this way (RunsByLane).

// The options of a batch tile of `lanes` lanes: pivot perturbation off, matching off
// (CUDSS_MATCHING_ALG_NONE, set explicitly: cuDSS's default matches general matrices),
// reordering, iterative refinement, the diagnostics the environment asks for, and, for a uniform
// batch, the lane count (CUDSS_CONFIG_UBATCH_SIZE; a stacked tile is one system). Hybrid memory
// and deterministic mode never reach a tile (RunsByLane). A suspect lane is factored again on a
// single side that matches (see "Batch sides").
ffi::Error ConfigureBatch(const Analysis& a, const Slot& slot, int64_t lanes,
                          cudssConfig_t config) {
    CONTRACT_TRY(ApplyDiagnosticConfig(slot.handle, config));
    const double epsilon = 0.0;
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_EPSILON, &epsilon, sizeof(epsilon)),
                   "cudssConfigSet pivot_epsilon of a batched factor");
    int matching = CUDSS_MATCHING_ALG_NONE;
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_MATCHING_ALG, &matching, sizeof(matching)),
                   "cudssConfigSet matching of a batched factor");
    if (a.reorder_alg >= 0) {
        int alg = a.reorder_alg;
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)),
                       "cudssConfigSet reordering of a batched factor");
    }
    int ir = RefinementSteps(a, slot);
    CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                   "cudssConfigSet ir_nsteps of a batched factor");
    CONTRACT_TRY(ConfigurePrecision(a, slot, config));
    if (a.batch_layout == kBatchUniform) {
        int size = static_cast<int>(lanes);
        CONTRACT_CUDSS(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE, &size, sizeof(size)),
                       "cudssConfigSet ubatch_size of a batched factor");
    }
    return ffi::Error::Success();
}

// Set the mask of `tile` (CUDSS_DATA_UBATCH_MASK) to the lanes `on` marks (one byte per lane of
// the batch side, the tile's lanes from its first on; every lane when null), unless it says so
// already.
ffi::Error SetTileMask(const Slot& slot, BatchTile& tile, const std::vector<uint8_t>* on) {
    bool same = true;
    for (int64_t i = 0; i < tile.lanes; ++i) {
        const int32_t word = on == nullptr || (*on)[static_cast<size_t>(tile.first + i)] != 0;
        same = same && tile.mask[static_cast<size_t>(i)] == word;
        tile.mask[static_cast<size_t>(i)] = word;
    }
    if (same) return ffi::Error::Success();
    CONTRACT_CUDSS(cudssDataSet(slot.handle, tile.data, CUDSS_DATA_UBATCH_MASK, tile.mask.data(),
                                tile.mask.size() * sizeof(int32_t)),
                   "cudssDataSet ubatch_mask of a batched factor");
    return ffi::Error::Success();
}

ffi::Error SetTileSolveMode(BatchTile& tile, cudssSolveMode_t mode) {
    if (tile.solve_mode == mode) return ffi::Error::Success();
    CONTRACT_CUDSS(cudssConfigSet(tile.config, CUDSS_CONFIG_SOLVE_MODE, &mode, sizeof(mode)),
                   "cudssConfigSet solve_mode of a batched factor");
    tile.solve_mode = mode;
    return ffi::Error::Success();
}

// Run the analysis phase of `tile` on a new data object, waited for, after setting its mask
// (every lane; cuDSS honours a mask only on a data object analysed with one), and, when `seed` is
// set, the seed of sides without matching (SetSeed). `status` receives the first status other
// than success, `seconds` the wall time of the phase and `with_tree` whether the seed took the
// partition tree.
ffi::Error RunTileAnalysis(const Slot& slot, BatchTile& tile, cudaStream_t stream, bool seed,
                           cudssStatus_t* status, double* seconds, bool* with_tree) {
    *with_tree = false;
    CONTRACT_CUDSS(cudssDataCreate(slot.handle, &tile.data), "cudssDataCreate of a batched factor");
    std::fill(tile.mask.begin(), tile.mask.end(), 1);
    *status = cudssDataSet(slot.handle, tile.data, CUDSS_DATA_UBATCH_MASK, tile.mask.data(),
                           tile.mask.size() * sizeof(int32_t));
    if (*status == CUDSS_STATUS_SUCCESS && seed) {
        *status = SetSeed(*slot.analysis, slot, tile.data, with_tree);
    }
    if (*status != CUDSS_STATUS_SUCCESS) return ffi::Error::Success();
    const auto start = std::chrono::steady_clock::now();
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, tile.config, tile.data,
                                 tile.A, tile.x_scratch, tile.b_scratch, status));
    if (*status == CUDSS_STATUS_SUCCESS) {
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
        *seconds = Seconds(start);
    }
    return ffi::Error::Success();
}

// A uniform-batch tile of `lanes` lanes of `side` from lane `first` on: its configuration, its
// matrices over the side's value copy and scratch vectors, and one cuDSS analysis of the batch,
// waited for.
// With seeding on, the tile is analysed in the ordering of the seed of sides without matching
// (the kept permutation and nested-dissection partition tree, as AnalyzeSide passes them); while
// none is kept the tile runs cold and keeps its own. A refused seed falls back to a cold
// analysis. The permutation and tree together reproduce the cold analysis's fill and schedule in
// a fraction of its time: on a 90,000-row convection-diffusion pattern of 16 lanes the seeded
// analysis took 19.7 ms against 208 ms cold, with the same fill per lane (4,580,048) and the same
// factorization time (84 ms), while the permutation alone, or the pattern permuted on the host
// and analysed without reordering (CUDSS_REORDERING_ALG_NONE), analysed in 202 to 212 ms and
// factored in 611 to 618 ms (measured on an RTX 4070 Ti with cuDSS 1.0.0.35). The fill per lane
// is recorded on the analysis (lu_nnz_batch) from the analysis phase.
ffi::Error CreateTile(Registry& reg, Slot& slot, BatchSide& side, int64_t first, int64_t lanes,
                      cudaStream_t stream, std::unique_ptr<BatchTile>* out) {
    Analysis& a = *slot.analysis;
    const size_t vs = slot.value_size;
    const size_t n = static_cast<size_t>(a.n);
    auto tile = std::make_unique<BatchTile>();
    tile->first = first;
    tile->lanes = lanes;
    tile->handle = slot.handle;
    tile->device = slot.device;
    tile->mask.assign(static_cast<size_t>(lanes), 1);
    CONTRACT_CUDSS(cudssConfigCreate(&tile->config), "cudssConfigCreate of a batched factor");
    CONTRACT_TRY(ConfigureBatch(a, slot, lanes, tile->config));
    if (Downcast(a, slot)) ++reg.stats.downcast_configurations;
    const DevicePattern& pattern = a.On(slot.device);
    char* values = static_cast<char*>(side.values) +
                   static_cast<size_t>(first) * static_cast<size_t>(a.nse) * vs;
    char* scratch_b = static_cast<char*>(side.scratch_b) + static_cast<size_t>(first) * n * vs;
    char* scratch_x = static_cast<char*>(side.scratch_x) + static_cast<size_t>(first) * n * vs;
    CONTRACT_CUDSS(cudssMatrixCreateCsr(&tile->A, a.n, a.n, a.nse, pattern.offsets, nullptr,
                                        pattern.columns, values, CUDA_R_32I, CUDA_R_32I,
                                        slot.dtype, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                        CUDSS_BASE_ZERO),
                   "cudssMatrixCreateCsr of a batched factor");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&tile->b_scratch, a.n, 1, a.n, scratch_b, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's scratch vectors");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&tile->x_scratch, a.n, 1, a.n, scratch_x, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's scratch vectors");
    bool seeded = false;
    double seconds = 0.0;
    if (reg.seeding && static_cast<int32_t>(a.tree_permutation.size()) == a.n) {
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;
        bool with_tree = false;
        CONTRACT_TRY(RunTileAnalysis(slot, *tile, stream, true, &status, &seconds, &with_tree));
        if (status == CUDSS_STATUS_SUCCESS) {
            seeded = true;
            ++a.seeded_analyses;
            ++reg.stats.seeded_analyses;
            if (with_tree) ++a.tree_seeded_analyses;
        } else {
            // cuDSS refused the seed: run a cold analysis on a fresh data object. Once its cleanup
            // has begun a data object is never used or destroyed again.
            ++a.seed_refusals;
            cudssDataDestroy(slot.handle, tile->data);
            tile->data = nullptr;
        }
    }
    if (!seeded) {
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;
        bool with_tree = false;
        CONTRACT_TRY(RunTileAnalysis(slot, *tile, stream, false, &status, &seconds, &with_tree));
        CONTRACT_CUDSS(status, "cudssExecute analysis of a batched factor");
        ++a.cold_analyses;
        ++reg.stats.cold_analyses;
        KeepColdOrdering(a, slot, tile->data, false);
    }
    a.analysis_seconds_batch_last = seconds;
    ++a.batch_analyses;
    ++reg.stats.batch_analyses;
    a.batch_seeded = seeded ? 1 : 0;
    const FactorFacts facts = QueryFactorFacts(slot.handle, tile->data);
    if (facts.lu_nnz > 0) a.lu_nnz_batch = facts.lu_nnz;
    int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
    QueryMemoryEstimates(slot.handle, tile->data, estimates);
    tile->permanent_device_bytes = std::max<int64_t>(estimates[0], 0);
    *out = std::move(tile);
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Stacked batch tiles
// --------------------------------------------------------------------------

// The partition tree cuDSS reports after an analysis (CUDSS_DATA_ND_PARTITION_TREE) and takes as
// a seed (CUDSS_DATA_USER_ND_PARTITION_TREE), as measured with cuDSS 1.0.0.35: kTreeNodes int32
// node sizes of a complete binary tree of kTreeLevels levels, the deepest level first and the
// root last, the nodes of each level left to right, a node of size zero empty; the permutation
// of the same analysis lists the rows of the nodes in that order. A tree of any other size, or
// whose sizes do not add up to the rows, is not used to seed a stacked tile.
constexpr int kTreeLevels = 10;
constexpr int kTreeNodes = (1 << kTreeLevels) - 1;

// The position in the tree array of node `j` of level `t` (the root is level 0).
inline int64_t TreeIndex(int t, int64_t j) {
    return (int64_t{1} << kTreeLevels) - (int64_t{1} << (t + 1)) + j;
}

// The seed of a stacked tile of `lanes` lanes, built from the seed of sides without matching
// (one lane's permutation and partition tree): the top k levels of the stacked tree, for the
// smallest k with 2^k >= lanes, are empty separators, and the subtree under node l of level k is
// lane l's tree (empty for l >= lanes) cut to the remaining levels: a node of the deepest kept
// level takes the rows of its whole subtree in post order (each child's subtree, then its own
// rows), which keeps the nested-dissection order of that subtree. Lane l's rows are shifted by
// l * n. Returns false when no seed is kept, its tree is not in the measured layout, or the lanes
// need every level (more than 2^(kTreeLevels - 1)); the tile then runs a cold analysis.
//
// On a 10,000-row convection-diffusion pattern of 16 lanes (RTX 4070 Ti, cuDSS 1.0.0.35) the
// seeded analysis took 27 ms against 333 ms cold, with the fill per lane of one lane analysed
// alone (380,100 against 377,561 cold) and a factorization of 4.1 against 4.7 ms; the
// permutation alone, without the tree, factored in 54 ms.
bool StackedSeed(const Analysis& a, int64_t lanes, std::vector<int32_t>* perm,
                 std::vector<int32_t>* tree) {
    const int64_t n = a.n;
    if (static_cast<int64_t>(a.tree_permutation.size()) != n ||
        a.nd_tree.size() != static_cast<size_t>(kTreeNodes) * sizeof(int32_t)) {
        return false;
    }
    int k = 0;
    while ((int64_t{1} << k) < lanes) ++k;
    if (k >= kTreeLevels) return false;
    std::vector<int32_t> sizes(static_cast<size_t>(kTreeNodes));
    std::memcpy(sizes.data(), a.nd_tree.data(), a.nd_tree.size());
    std::vector<int64_t> start(static_cast<size_t>(kTreeNodes) + 1, 0);
    for (int i = 0; i < kTreeNodes; ++i) {
        if (sizes[i] < 0) return false;
        start[i + 1] = start[i] + sizes[i];
    }
    if (start[kTreeNodes] != n) return false;
    const int cut = kTreeLevels - 1 - k;  // the deepest level of a lane's tree that is kept
    // One lane's rows per kept node.
    std::vector<std::vector<int32_t>> rows(static_cast<size_t>(kTreeNodes));
    std::function<void(int, int64_t, std::vector<int32_t>&)> post =
        [&](int t, int64_t j, std::vector<int32_t>& out) {
            if (t + 1 < kTreeLevels) {
                post(t + 1, 2 * j, out);
                post(t + 1, 2 * j + 1, out);
            }
            const int64_t i = TreeIndex(t, j);
            out.insert(out.end(), a.tree_permutation.begin() + start[i],
                       a.tree_permutation.begin() + start[i + 1]);
        };
    for (int t = 0; t <= cut; ++t) {
        for (int64_t j = 0; j < (int64_t{1} << t); ++j) {
            std::vector<int32_t>& node = rows[static_cast<size_t>(TreeIndex(t, j))];
            if (t < cut) {
                const int64_t i = TreeIndex(t, j);
                node.assign(a.tree_permutation.begin() + start[i],
                            a.tree_permutation.begin() + start[i + 1]);
            } else {
                post(t, j, node);
            }
        }
    }
    tree->assign(static_cast<size_t>(kTreeNodes), 0);
    perm->clear();
    perm->reserve(static_cast<size_t>(lanes * n));
    for (int t = kTreeLevels - 1; t >= 0; --t) {
        for (int64_t j = 0; j < (int64_t{1} << t); ++j) {
            if (t < k) continue;  // an empty separator between lanes
            const int lane_level = t - k;
            const int64_t lane = j >> lane_level;
            if (lane >= lanes) continue;  // an empty lane subtree
            const std::vector<int32_t>& node =
                rows[static_cast<size_t>(TreeIndex(lane_level, j - (lane << lane_level)))];
            (*tree)[static_cast<size_t>(TreeIndex(t, j))] = static_cast<int32_t>(node.size());
            const int32_t shift = static_cast<int32_t>(lane * n);
            for (int32_t row : node) perm->push_back(row + shift);
        }
    }
    return static_cast<int64_t>(perm->size()) == lanes * n;
}

// The cuDSS objects of a one-lane analysis that keeps a seed, destroyed with it.
struct LaneSeedObjects {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrix_t x = nullptr;

    ~LaneSeedObjects() {
        if (A != nullptr) cudssMatrixDestroy(A);
        if (b != nullptr) cudssMatrixDestroy(b);
        if (x != nullptr) cudssMatrixDestroy(x);
        if (data != nullptr) cudssDataDestroy(handle, data);
        if (config != nullptr) cudssConfigDestroy(config);
    }
};

// With seeding on and no seed of sides without matching kept yet: run the cold analysis of lane
// 0 of `side` alone, with a batch tile's options, on a data object of its own, waited for, and
// keep its permutation and partition tree as that seed (KeepColdOrdering), so the side's stacked
// tiles are analysed seeded. Counted as a cold analysis. Its cost is one lane's analysis (50 ms
// for a 10,000-row convection-diffusion pattern, 25 ms for a 961-row battery Jacobian; RTX 4070
// Ti, cuDSS 1.0.0.35).
ffi::Error EnsureLaneSeed(Registry& reg, Slot& slot, BatchSide& side, cudaStream_t stream) {
    Analysis& a = *slot.analysis;
    if (!reg.seeding || static_cast<int32_t>(a.tree_permutation.size()) == a.n) {
        return ffi::Error::Success();
    }
    const DevicePattern& pattern = a.On(slot.device);
    LaneSeedObjects o;
    o.handle = slot.handle;
    CONTRACT_CUDSS(cudssConfigCreate(&o.config), "cudssConfigCreate of a batched factor's seed");
    CONTRACT_TRY(ConfigureBatch(a, slot, 1, o.config));
    CONTRACT_CUDSS(cudssMatrixCreateCsr(&o.A, a.n, a.n, a.nse, pattern.offsets, nullptr,
                                        pattern.columns, side.values, CUDA_R_32I, CUDA_R_32I,
                                        slot.dtype, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                        CUDSS_BASE_ZERO),
                   "cudssMatrixCreateCsr of a batched factor's seed");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&o.b, a.n, 1, a.n, side.scratch_b, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's seed");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&o.x, a.n, 1, a.n, side.scratch_x, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's seed");
    CONTRACT_CUDSS(cudssDataCreate(slot.handle, &o.data),
                   "cudssDataCreate of a batched factor's seed");
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, o.config, o.data, o.A, o.x,
                                 o.b, &status));
    CONTRACT_CUDSS(status, "cudssExecute analysis of a batched factor's seed");
    CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
    ++a.cold_analyses;
    ++reg.stats.cold_analyses;
    KeepColdOrdering(a, slot, o.data, false);
    return ffi::Error::Success();
}

// A stacked tile of `lanes` lanes of `side` from lane `first` on: its configuration, its
// matrices (lanes * n rows over the side's stacked pattern, value copy and scratch vectors) and
// one cuDSS analysis, waited for. With seeding on, the analysis is seeded with the stacked
// permutation and partition tree that StackedSeed builds from one lane's seed (kept first by
// EnsureLaneSeed when none is); otherwise, or when the seed cannot be built or cuDSS refuses it,
// it runs cold, which costs about the lanes times one lane's analysis. The fill per lane is
// recorded on the analysis (lu_nnz_batch, rounded up).
ffi::Error CreateStackedTile(Registry& reg, Slot& slot, BatchSide& side, int64_t first,
                             int64_t lanes, cudaStream_t stream,
                             std::unique_ptr<BatchTile>* out) {
    Analysis& a = *slot.analysis;
    const size_t vs = slot.value_size;
    const size_t n = static_cast<size_t>(a.n);
    const int64_t rows = lanes * a.n;
    const int64_t entries = lanes * a.nse;
    auto tile = std::make_unique<BatchTile>();
    tile->first = first;
    tile->lanes = lanes;
    tile->stacked = true;
    tile->handle = slot.handle;
    tile->device = slot.device;
    CONTRACT_CUDSS(cudssConfigCreate(&tile->config), "cudssConfigCreate of a batched factor");
    CONTRACT_TRY(ConfigureBatch(a, slot, lanes, tile->config));
    if (Downcast(a, slot)) ++reg.stats.downcast_configurations;
    char* values = static_cast<char*>(side.values) +
                   static_cast<size_t>(first) * static_cast<size_t>(a.nse) * vs;
    char* scratch_b = static_cast<char*>(side.scratch_b) + static_cast<size_t>(first) * n * vs;
    char* scratch_x = static_cast<char*>(side.scratch_x) + static_cast<size_t>(first) * n * vs;
    CONTRACT_CUDSS(cudssMatrixCreateCsr(&tile->A, rows, rows, entries, side.stacked_offsets,
                                        nullptr, side.stacked_columns, values, CUDA_R_32I,
                                        CUDA_R_32I, slot.dtype, CUDSS_MTYPE_GENERAL,
                                        CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                   "cudssMatrixCreateCsr of a batched factor");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&tile->b_scratch, rows, 1, rows, scratch_b, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's scratch vectors");
    CONTRACT_CUDSS(cudssMatrixCreateDn(&tile->x_scratch, rows, 1, rows, scratch_x, slot.dtype,
                                       CUDSS_LAYOUT_COL_MAJOR),
                   "cudssMatrixCreateDn of a batched factor's scratch vectors");
    CONTRACT_TRY(EnsureLaneSeed(reg, slot, side, stream));
    bool seeded = false;
    double seconds = 0.0;
    std::vector<int32_t> perm;
    std::vector<int32_t> tree;
    if (reg.seeding && StackedSeed(a, lanes, &perm, &tree)) {
        CONTRACT_CUDSS(cudssDataCreate(slot.handle, &tile->data),
                       "cudssDataCreate of a batched factor");
        cudssStatus_t status = cudssDataSet(slot.handle, tile->data, CUDSS_DATA_USER_PERM,
                                            perm.data(), perm.size() * sizeof(int32_t));
        if (status == CUDSS_STATUS_SUCCESS) {
            status = cudssDataSet(slot.handle, tile->data, CUDSS_DATA_USER_ND_PARTITION_TREE,
                                  tree.data(), tree.size() * sizeof(int32_t));
        }
        if (status == CUDSS_STATUS_SUCCESS) {
            const auto start = std::chrono::steady_clock::now();
            CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, tile->config,
                                         tile->data, tile->A, tile->x_scratch, tile->b_scratch,
                                         &status));
            if (status == CUDSS_STATUS_SUCCESS) {
                CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
                seconds = Seconds(start);
            }
        }
        if (status == CUDSS_STATUS_SUCCESS) {
            seeded = true;
            ++a.seeded_analyses;
            ++reg.stats.seeded_analyses;
            ++a.tree_seeded_analyses;
        } else {
            // cuDSS refused the seed: run a cold analysis on a fresh data object.
            ++a.seed_refusals;
            cudssDataDestroy(slot.handle, tile->data);
            tile->data = nullptr;
        }
    }
    if (!seeded) {
        CONTRACT_CUDSS(cudssDataCreate(slot.handle, &tile->data),
                       "cudssDataCreate of a batched factor");
        cudssStatus_t status = CUDSS_STATUS_SUCCESS;
        const auto start = std::chrono::steady_clock::now();
        CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_ANALYSIS, tile->config,
                                     tile->data, tile->A, tile->x_scratch, tile->b_scratch,
                                     &status));
        CONTRACT_CUDSS(status, "cudssExecute analysis of a batched factor");
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (analysis)");
        seconds = Seconds(start);
        ++a.cold_analyses;
        ++reg.stats.cold_analyses;
    }
    a.analysis_seconds_batch_last = seconds;
    ++a.batch_analyses;
    ++reg.stats.batch_analyses;
    a.batch_seeded = seeded ? 1 : 0;
    const FactorFacts facts = QueryFactorFacts(slot.handle, tile->data);
    if (facts.lu_nnz > 0) a.lu_nnz_batch = (facts.lu_nnz + lanes - 1) / lanes;
    int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
    QueryMemoryEstimates(slot.handle, tile->data, estimates);
    tile->permanent_device_bytes = std::max<int64_t>(estimates[0], 0);
    *out = std::move(tile);
    return ffi::Error::Success();
}

// The stacked pattern of `side` for tiles of up to `lanes` lanes, built on the device
// (QueueStackedPattern) from the analysis's pattern on the slot's device.
ffi::Error EnsureStackedPattern(Slot& slot, BatchSide& side, int64_t lanes, cudaStream_t stream) {
    const Analysis& a = *slot.analysis;
    const DevicePattern& pattern = a.On(slot.device);
    const size_t offsets = static_cast<size_t>(lanes * a.n + 1);
    const size_t columns = static_cast<size_t>(lanes * a.nse);
    CONTRACT_CUDA(cudaMalloc(&side.stacked_offsets, offsets * sizeof(int32_t)),
                  "cudaMalloc (stacked batch pattern)");
    CONTRACT_CUDA(cudaMalloc(&side.stacked_columns, columns * sizeof(int32_t)),
                  "cudaMalloc (stacked batch pattern)");
    side.stacked_pattern_bytes = static_cast<int64_t>((offsets + columns) * sizeof(int32_t));
    CONTRACT_CUDA(diff_soup_assembly::QueueStackedPattern(stream, pattern.offsets, pattern.columns,
                                                          a.n, a.nse, lanes, side.stacked_offsets,
                                                          side.stacked_columns),
                  "stacked pattern kernel");
    return ffi::Error::Success();
}

// The batch side of `slot`, created on first use, its value copy filled with the values of
// `fill` (every lane), and its tiles analysed (none for a batch that runs lane by lane). The tiles
// take as many lanes as TileLanes allows for the fill per lane known when they are made; when the
// first tile's analysis reports a fill per lane that allows fewer, the tiles are made again for
// it.
ffi::Error EnsureBatchSide(Registry& reg, Slot& slot, const ValueFill& fill, cudaStream_t stream,
                           BatchSide** out) {
    std::unique_ptr<BatchSide>& holder = slot.batch_primary;
    if (holder) {
        *out = holder.get();
        return ffi::Error::Success();
    }
    Analysis& a = *slot.analysis;
    CONTRACT_TRY(EnsurePattern(reg, a, slot.device, stream));
    const int64_t lanes = slot.lanes;
    const size_t vs = slot.value_size;
    auto side = std::make_unique<BatchSide>();
    side->handle = slot.handle;
    side->device = slot.device;
    side->lanes = lanes;
    side->by_lane = RunsByLane(a);
    side->status.assign(static_cast<size_t>(lanes), kSingular);
    side->generations.assign(static_cast<size_t>(lanes), 0);
    side->rescued.assign(static_cast<size_t>(lanes), 0);
    side->factored.assign(static_cast<size_t>(lanes), 0);
    side->rescue.resize(static_cast<size_t>(lanes));
    const size_t vector_bytes = static_cast<size_t>(lanes) * static_cast<size_t>(a.n) * vs;
    side->value_bytes = lanes * a.nse * static_cast<int64_t>(vs);
    side->scratch_bytes = 2 * static_cast<int64_t>(vector_bytes);
    CONTRACT_CUDA(cudaMalloc(&side->values, static_cast<size_t>(side->value_bytes)),
                  "cudaMalloc (batched factor value copies)");
    CONTRACT_CUDA(cudaMalloc(&side->scratch_b, vector_bytes),
                  "cudaMalloc (batched factor scratch)");
    CONTRACT_CUDA(cudaMalloc(&side->scratch_x, vector_bytes),
                  "cudaMalloc (batched factor scratch)");
    CONTRACT_CUDA(cudaMemsetAsync(side->scratch_b, 0, vector_bytes, stream),
                  "cudaMemsetAsync (batched factor scratch)");
    CONTRACT_CUDA(cudaMemsetAsync(side->scratch_x, 0, vector_bytes, stream),
                  "cudaMemsetAsync (batched factor scratch)");
    CONTRACT_TRY(FillValues(reg, slot, fill, side->values, lanes, stream));
    if (!side->by_lane) {
        int64_t per_tile = TileLanes(reg, a, LaneFactorEntries(a));
        side->stacked = a.batch_layout == kBatchStacked;
        // A tile of fewer lanes uses the leading part of the largest tile's stacked pattern.
        if (side->stacked) {
            CONTRACT_TRY(EnsureStackedPattern(slot, *side, std::min(per_tile, lanes), stream));
        }
        for (int64_t first = 0; first < lanes;) {
            std::unique_ptr<BatchTile> tile;
            const int64_t tile_lanes = std::min(per_tile, lanes - first);
            if (side->stacked) {
                CONTRACT_TRY(CreateStackedTile(reg, slot, *side, first, tile_lanes, stream, &tile));
            } else {
                CONTRACT_TRY(CreateTile(reg, slot, *side, first, tile_lanes, stream, &tile));
            }
            const int64_t fits = TileLanes(reg, a, LaneFactorEntries(a));
            if (tile->lanes > fits) {
                // The fill the analysis reported allows fewer lanes per tile (fits is below every
                // earlier tile size, so this ends).
                tile.reset();
                side->tiles.clear();
                per_tile = fits;
                first = 0;
                continue;
            }
            first += tile->lanes;
            side->tiles.push_back(std::move(tile));
        }
    }
    holder = std::move(side);
    *out = holder.get();
    return ffi::Error::Success();
}

// The lanes of `tile` that `on` marks.
int64_t TileCount(const BatchTile& tile, const std::vector<uint8_t>& on) {
    int64_t count = 0;
    for (int64_t i = 0; i < tile.lanes; ++i) count += on[static_cast<size_t>(tile.first + i)] != 0;
    return count;
}

// Run one numeric phase (factorization, or refactorization when `refactor`) on the lanes of
// `tile` that `on` marks (`count` of them; CUDSS_DATA_UBATCH_MASK), over the values in the side's
// value copy, for a call on `stream`. Every other lane keeps its factor.
ffi::Error TileNumericPhase(Registry& reg, const Slot& slot, BatchTile& tile,
                            const std::vector<uint8_t>& on, int64_t count, bool refactor,
                            cudaStream_t stream) {
    const cudssPhase_t phase = refactor ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
    if (count < tile.lanes && Downcast(*slot.analysis, slot)) {
        return ffi::Error(
            ffi::ErrorCode::kFailedPrecondition,
            "cuDSS cannot factor some lanes of a uniform batch (batch_layout=\"uniform\") "
            "whose float64 factors are stored downcast (downcast_factors): refactor every lane "
            "of the batch at once, or analyze with batch_layout=\"stacked\" (the default), or "
            "with downcast_factors=False or left out, which keeps the factors of a batch in "
            "float64.");
    }
    CONTRACT_TRY(SetTileMask(slot, tile, &on));
    CONTRACT_TRY(SetTileSolveMode(tile, CUDSS_SOLVE_MODE_NORMAL));
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, phase, tile.config, tile.data, tile.A,
                                 tile.x_scratch, tile.b_scratch, &status));
    if (status != CUDSS_STATUS_SUCCESS) {
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, tile.data, estimates);
        return FactorizationFailed(*slot.analysis, status, refactor, estimates[1]);
    }
    ++reg.stats.batched_factorizations;
    reg.stats.lanes_factored += count;
    return ffi::Error::Success();
}

// The status solve of the lanes `on` marks (see "Batch sides"): the right-hand sides
// b_l = A_l x* of every lane into the side's scratch, one masked solve per tile that holds such
// a lane into the side's scratch solution, the errors x_l - x*, and one scan of every lane's
// values and errors read back with one wait into `scan` (whether the values and the errors are
// finite, and the largest error magnitude). The status solve runs without iterative refinement,
// which would double its cost (2.0 against 1.0 ms for 16 lanes of 10,000 rows, measured on an
// RTX 4070 Ti with cuDSS 1.0.0.35) and only shrinks errors that already lie far below the suspect
// bound; a downcast factor's status solve refines as its solves do (StatusRefinementSteps).
ffi::Error StatusSolve(Registry& reg, Slot& slot, BatchSide& side, const std::vector<uint8_t>& on,
                       cudaStream_t stream, std::vector<LaneScan>* scan) {
    Analysis& a = *slot.analysis;
    const DevicePattern& pattern = a.On(slot.device);
    const bool f64 = slot.dtype == CUDA_R_64F;
    CONTRACT_CUDA(diff_soup_status::QueueProbeRightHandSides(stream, f64, pattern.offsets,
                                                             pattern.columns, side.values, a.n,
                                                             a.nse, side.lanes, side.scratch_b),
                  "status solve right-hand-side kernel");
    for (auto& tile_holder : side.tiles) {
        BatchTile& tile = *tile_holder;
        if (TileCount(tile, on) == 0) continue;
        CONTRACT_TRY(SetTileMask(slot, tile, &on));
        CONTRACT_TRY(SetTileSolveMode(tile, CUDSS_SOLVE_MODE_NORMAL));
        int ir = StatusRefinementSteps(a, slot);
        const int ir_steps = RefinementSteps(a, slot);
        if (ir_steps != ir) {
            CONTRACT_CUDSS(cudssConfigSet(tile.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                           "cudssConfigSet ir_nsteps of a batched factor's status solve");
        }
        cudssStatus_t solved = CUDSS_STATUS_SUCCESS;
        CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_SOLVE, tile.config, tile.data,
                                     tile.A, tile.x_scratch, tile.b_scratch, &solved));
        if (ir_steps != ir) {
            ir = ir_steps;
            CONTRACT_CUDSS(cudssConfigSet(tile.config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                           "cudssConfigSet ir_nsteps of a batched factor");
        }
        CONTRACT_CUDSS(solved, "cudssExecute status solve of a batched factor");
        ++reg.stats.status_solves;
    }
    CONTRACT_CUDA(diff_soup_status::QueueProbeErrors(stream, f64, side.scratch_x, a.n, side.lanes),
                  "status solve error kernel");
    return ScanLanes(reg, slot.device, stream, slot.dtype, side.values, a.nse, side.scratch_x, a.n,
                     side.lanes, scan);
}

// Run one numeric phase (factorization, or refactorization when `refactor`) on every lane of the
// stacked tile `tile` (`count` of them asked by the request; the others are factored again from
// the values their copies hold), then copy the factor's diagonal (CUDSS_DATA_DIAG, every lane's n
// entries in row order, lane after lane) into the side's scratch solution vectors at the tile's
// lanes. cuDSS reports the diagonal of a stacked factor in the rows' original order, lane l at
// rows [l * n, (l + 1) * n) (measured with cuDSS 1.0.0.35: lanes that are powers of two times
// one lane gave exactly those multiples of its diagonal, seeded and cold).
ffi::Error StackedTilePhase(Registry& reg, const Slot& slot, BatchSide& side, BatchTile& tile,
                            int64_t count, bool refactor, cudaStream_t stream) {
    const cudssPhase_t phase = refactor ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION;
    CONTRACT_TRY(SetTileSolveMode(tile, CUDSS_SOLVE_MODE_NORMAL));
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, phase, tile.config, tile.data, tile.A,
                                 tile.x_scratch, tile.b_scratch, &status));
    if (status != CUDSS_STATUS_SUCCESS) {
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, tile.data, estimates);
        return FactorizationFailed(*slot.analysis, status, refactor, estimates[1]);
    }
    ++reg.stats.batched_factorizations;
    reg.stats.lanes_factored += tile.lanes;
    reg.stats.lanes_factored_unasked += tile.lanes - count;
    const size_t lane_bytes = static_cast<size_t>(slot.analysis->n) * slot.value_size;
    const size_t bytes = static_cast<size_t>(tile.lanes) * lane_bytes;
    void* target =
        static_cast<char*>(side.scratch_x) + static_cast<size_t>(tile.first) * lane_bytes;
    size_t written = 0;
    CONTRACT_TRY(HandleEnter(slot, stream));
    const cudssStatus_t read =
        cudssDataGet(slot.handle, tile.data, CUDSS_DATA_DIAG, target, bytes, &written);
    CONTRACT_TRY(HandleLeave(slot, stream));
    CONTRACT_CUDSS(read, "cudssDataGet diag of a batched factor");
    if (written != bytes) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          "cuDSS wrote " + std::to_string(written) +
                              " bytes for the diagonals of a batched factor's " +
                              std::to_string(tile.lanes) + " lanes; they take " +
                              std::to_string(bytes) + " bytes.");
    }
    return ffi::Error::Success();
}

// The numeric phases and lane statuses of a stacked batch side, for FactorBatchSide (whose
// value copy and marks it takes as they stand): each tile that holds a lane `on` marks runs one
// numeric phase over all its lanes (a stacked tile has no mask), the refactorization when
// `refactorization` asks for it and the tile holds a factorization of every lane, the
// factorization otherwise. A lane the request did not mark is factored from the values its copy
// holds, which are the values of its factor, so it keeps its factor's matrix, its status and its
// generation. One device pass over the lanes' values and diagonals, read back once, then
// settles the marked lanes by the rule the factor diagonal gives a batch: non-finite values are
// NONFINITE; a diagonal with a non-finite or a zero entry, or whose smallest magnitude is at most
// the square root of the unit roundoff times its largest (BatchSuspectRatio), is suspect; every
// other marked lane is SUCCESS. An unmarked lane that its tile served with SUCCESS is checked by
// the same rule and treated as suspect if it fails it. A tile whose refactorization left a lane
// suspect is factored afresh and scanned again, and a lane still suspect is factored again alone
// on its rescue side, whose status it takes. No status solve runs.
ffi::Error FactorStackedSide(Registry& reg, Slot& slot, BatchSide& side,
                             const std::vector<uint8_t>& on, bool refactorization,
                             cudaStream_t stream, int64_t* phases) {
    Analysis& a = *slot.analysis;
    const int64_t lanes = side.lanes;
    const size_t vs = slot.value_size;
    const size_t lane_bytes = static_cast<size_t>(a.nse) * vs;
    std::vector<uint8_t> ran(static_cast<size_t>(lanes), 0);
    std::vector<uint8_t> refactored_tile(side.tiles.size(), 0);
    for (size_t t = 0; t < side.tiles.size(); ++t) {
        BatchTile& tile = *side.tiles[t];
        const int64_t count = TileCount(tile, on);
        if (count == 0) continue;
        bool refactor = refactorization;
        for (int64_t i = 0; i < tile.lanes; ++i) {
            refactor = refactor && side.factored[tile.first + i] != 0;
        }
        CONTRACT_TRY(StackedTilePhase(reg, slot, side, tile, count, refactor, stream));
        ++*phases;
        refactored_tile[t] = refactor ? 1 : 0;
        for (int64_t i = 0; i < tile.lanes; ++i) {
            side.factored[tile.first + i] = 1;
            ran[tile.first + i] = 1;
        }
    }
    std::vector<LaneScan> scan;
    CONTRACT_TRY(ScanLanes(reg, slot.device, stream, slot.dtype, side.values, a.nse,
                           side.scratch_x, a.n, lanes, &scan));
    std::vector<uint8_t> finite(static_cast<size_t>(lanes), 1);
    for (int64_t l = 0; l < lanes; ++l) {
        if (!on[l] || scan[l].values_finite) continue;
        finite[l] = 0;
        side.status[l] = kNonfinite;
        ++reg.stats.nonfinite;
    }
    const double threshold = BatchSuspectRatio(slot.dtype);
    // An unmarked lane is checked when its tile ran and the tile's factor served it.
    std::vector<uint8_t> checked(static_cast<size_t>(lanes), 0);
    for (int64_t l = 0; l < lanes; ++l) {
        checked[l] = ran[l] && (on[l] ? finite[l] != 0
                                      : side.status[l] == kSuccess && !side.rescued[l]);
    }
    const auto suspect = [&](int64_t l) {
        const LaneScan& d = scan[l];
        return checked[l] != 0 &&
               (!d.finite || d.smallest == 0.0 || d.smallest <= threshold * d.largest);
    };
    // A refactorization that left a lane suspect kept a pivot sequence that no longer serves:
    // that tile is factored afresh.
    for (size_t t = 0; t < side.tiles.size(); ++t) {
        if (!refactored_tile[t]) continue;
        BatchTile& tile = *side.tiles[t];
        bool any = false;
        for (int64_t i = 0; i < tile.lanes; ++i) any = any || suspect(tile.first + i);
        if (!any) continue;
        CONTRACT_TRY(StackedTilePhase(reg, slot, side, tile, TileCount(tile, on), false, stream));
        ++*phases;
        std::vector<LaneScan> again;
        CONTRACT_TRY(ScanLanes(reg, slot.device, stream, slot.dtype, nullptr, 0,
                               static_cast<char*>(side.scratch_x) +
                                   static_cast<size_t>(tile.first) * static_cast<size_t>(a.n) * vs,
                               a.n, tile.lanes, &again));
        for (int64_t i = 0; i < tile.lanes; ++i) {
            LaneScan& d = scan[tile.first + i];
            d.finite = again[i].finite;
            d.smallest = again[i].smallest;
            d.largest = again[i].largest;
        }
    }
    for (int64_t l = 0; l < lanes; ++l) {
        if (on[l] && finite[l] && !suspect(l)) side.status[l] = kSuccess;
    }
    for (int64_t l = 0; l < lanes; ++l) {
        if (!suspect(l)) continue;
        const void* lane_values = static_cast<const char*>(side.values) +
                                  static_cast<size_t>(l) * lane_bytes;
        std::unique_ptr<Side>& rescue = side.rescue[l];
        if (!rescue) {
            CONTRACT_TRY(CreateSide(reg, slot, ValueFill::Csr(lane_values), stream, &rescue));
        }
        rescue->generation = side.generations[l];
        bool rescue_ran = false;
        bool used_refactorization = false;
        // The batch reads its lanes' statuses on the host, so a rescue settles its verdict at once.
        CONTRACT_TRY(FactorSide(reg, slot, *rescue, ValueFill::Csr(lane_values), stream, false,
                                &rescue_ran, &used_refactorization, true));
        if (rescue_ran) ++reg.stats.lane_rescues;
        side.rescued[l] = 1;
        side.status[l] = rescue->status;
    }
    for (auto& tile_holder : side.tiles) {
        BatchTile& tile = *tile_holder;
        if (tile.facts_known || TileCount(tile, side.factored) == 0) continue;
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, tile.data, estimates);
        if (estimates[0] >= 0) tile.permanent_device_bytes = estimates[0];
        tile.facts_known = true;
    }
    return ffi::Error::Success();
}

// Point the solve-time matrices of a stacked tile at `b` and `x` (lanes * n rows and `nrhs`
// columns each, column-major), recreating them when the number of right-hand sides changed.
ffi::Error PointStackedTileAt(const Slot& slot, BatchTile& tile, const void* b, void* x,
                              int64_t nrhs) {
    const int64_t rows = slot.analysis->n * tile.lanes;
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (nrhs > limit / std::max<int64_t>(rows, 1)) {
        return InvalidCall("a batched solve of " + std::to_string(nrhs) +
                           " right-hand sides per lane over " + std::to_string(tile.lanes) +
                           " lanes of " + std::to_string(slot.analysis->n) +
                           " rows holds more than 2**31 - 1 entries per batch; solve at most " +
                           std::to_string(std::max<int64_t>(limit / rows, 1)) +
                           " right-hand sides per lane in one call.");
    }
    const bool recreate = tile.b == nullptr || tile.x == nullptr || tile.nrhs != nrhs;
    if (!recreate && tile.last_b == b && tile.last_x == x) return ffi::Error::Success();
    tile.last_b = nullptr;
    tile.last_x = nullptr;
    if (recreate) {
        if (tile.b != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(tile.b);
            tile.b = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy b of a batched factor");
        }
        if (tile.x != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(tile.x);
            tile.x = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy x of a batched factor");
        }
        CONTRACT_CUDSS(cudssMatrixCreateDn(&tile.b, rows, nrhs, rows, const_cast<void*>(b),
                                           slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
                       "cudssMatrixCreateDn b of a batched factor");
        CONTRACT_CUDSS(cudssMatrixCreateDn(&tile.x, rows, nrhs, rows, x, slot.dtype,
                                           CUDSS_LAYOUT_COL_MAJOR),
                       "cudssMatrixCreateDn x of a batched factor");
        tile.nrhs = nrhs;
    } else {
        CONTRACT_CUDSS(cudssMatrixSetValues(tile.b, const_cast<void*>(b)),
                       "cudssMatrixSetValues b of a batched factor");
        CONTRACT_CUDSS(cudssMatrixSetValues(tile.x, x),
                       "cudssMatrixSetValues x of a batched factor");
    }
    tile.last_b = b;
    tile.last_x = x;
    return ffi::Error::Success();
}

// Solve every lane of the stacked tile `tile` at once (a stacked factor has no mask; SolveBatch
// then overwrites the lanes its factor does not serve), for `nrhs` right-hand sides per lane
// (lane i of `b` and `x` at i * n * nrhs entries), with A_bc^T when `transpose`. With one
// right-hand side per lane the lanes of `b` and `x` already are the stacked vectors; with
// several, they are regrouped column by column into the side's stack buffers and back.
ffi::Error SolveStackedTile(Registry& reg, Slot& slot, BatchSide& side, BatchTile& tile,
                            const void* b, void* x, int64_t nrhs, cudaStream_t stream,
                            bool transpose) {
    const size_t vs = slot.value_size;
    const size_t n = static_cast<size_t>(slot.analysis->n);
    const size_t rows = n * static_cast<size_t>(tile.lanes);
    const size_t lane_bytes = n * static_cast<size_t>(nrhs) * vs;
    const char* lane_b = static_cast<const char*>(b) + static_cast<size_t>(tile.first) * lane_bytes;
    char* lane_x = static_cast<char*>(x) + static_cast<size_t>(tile.first) * lane_bytes;
    const void* tile_b = lane_b;
    void* tile_x = lane_x;
    if (nrhs > 1) {
        const size_t bytes = static_cast<size_t>(side.lanes) * lane_bytes;
        if (side.stack_bytes < bytes) {
            if (side.stack_b != nullptr) cudaFree(side.stack_b);
            if (side.stack_x != nullptr) cudaFree(side.stack_x);
            side.stack_b = nullptr;
            side.stack_x = nullptr;
            side.stack_bytes = 0;
            CONTRACT_CUDA(cudaMalloc(&side.stack_b, bytes), "cudaMalloc (stacked batch solve)");
            CONTRACT_CUDA(cudaMalloc(&side.stack_x, bytes), "cudaMalloc (stacked batch solve)");
            side.stack_bytes = bytes;
        }
        char* stack_b = static_cast<char*>(side.stack_b) +
                        static_cast<size_t>(tile.first) * lane_bytes;
        tile_b = stack_b;
        tile_x = static_cast<char*>(side.stack_x) + static_cast<size_t>(tile.first) * lane_bytes;
        for (int64_t j = 0; j < nrhs; ++j) {
            CONTRACT_CUDA(cudaMemcpy2DAsync(stack_b + static_cast<size_t>(j) * rows * vs, n * vs,
                                            lane_b + static_cast<size_t>(j) * n * vs, lane_bytes,
                                            n * vs, static_cast<size_t>(tile.lanes),
                                            cudaMemcpyDeviceToDevice, stream),
                          "cudaMemcpy2DAsync (stacked batch right-hand sides)");
        }
    }
    CONTRACT_TRY(PointStackedTileAt(slot, tile, tile_b, tile_x, nrhs));
    if (ZeroFills(*slot.analysis, slot)) {
        CONTRACT_TRY(ZeroSolveOutput(tile_x, rows * static_cast<size_t>(nrhs) * vs, stream, false));
        ++reg.stats.zero_fills;
    }
    CONTRACT_TRY(SetTileSolveMode(tile, transpose ? CUDSS_SOLVE_MODE_TRANSPOSE
                                                  : CUDSS_SOLVE_MODE_NORMAL));
    cudssStatus_t solved = CUDSS_STATUS_SUCCESS;
    CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_SOLVE, tile.config, tile.data, tile.A,
                                 tile.x, tile.b, &solved));
    CONTRACT_CUDSS(solved, transpose ? "cudssExecute transposed solve of a batched factor"
                                     : "cudssExecute solve of a batched factor");
    ++reg.stats.batched_solves;
    if (transpose) ++reg.stats.transposed_solves;
    if (nrhs > 1) {
        const char* stack_x = static_cast<const char*>(tile_x);
        for (int64_t j = 0; j < nrhs; ++j) {
            CONTRACT_CUDA(cudaMemcpy2DAsync(lane_x + static_cast<size_t>(j) * n * vs, lane_bytes,
                                            stack_x + static_cast<size_t>(j) * rows * vs, n * vs,
                                            n * vs, static_cast<size_t>(tile.lanes),
                                            cudaMemcpyDeviceToDevice, stream),
                          "cudaMemcpy2DAsync (stacked batch solutions)");
        }
    }
    return ffi::Error::Success();
}

// Factor the lanes of the batch side that `lanes_on` marks (every lane when null) and settle
// their statuses; every other lane keeps its factor and its status. `raw` (the call's values,
// every lane) is assembled into the marked lanes of the side's value copy first; null leaves the
// value copy as the caller arranged it. Each tile that holds a marked lane runs one numeric phase
// over its marked lanes (CUDSS_DATA_UBATCH_MASK): the refactorization when `refactorization`
// asks for it and the tile holds a factorization of every one of them, the factorization
// otherwise. A status solve then settles each marked lane: a lane with non-finite values is
// NONFINITE; a lane whose status solve error is non-finite, or larger than the square root of the
// unit roundoff times the largest entry of x*, is suspect. Suspect lanes that a refactorization
// left so are factored afresh (one masked factorization of those lanes per tile) and solved
// again. A lane still suspect is factored again alone on its rescue side (a single side with
// matching), whose status it takes; every other marked lane is SUCCESS. A batch that runs lane
// by lane factors each marked lane on its own single side instead (FactorSide, its verdict settled
// at once), whose status it takes. `phases` counts the numeric phases (of tiles, or of the lanes
// of a batch that runs lane by lane).
ffi::Error FactorBatchSide(Registry& reg, Slot& slot, BatchSide& side, const RawValues* raw,
                           cudaStream_t stream, bool refactorization,
                           const std::vector<uint8_t>* lanes_on, int64_t* phases) {
    Analysis& a = *slot.analysis;
    const int64_t lanes = side.lanes;
    const size_t lane_bytes = static_cast<size_t>(a.nse) * slot.value_size;
    *phases = 0;
    const std::vector<uint8_t> on =
        lanes_on != nullptr ? *lanes_on : std::vector<uint8_t>(static_cast<size_t>(lanes), 1);
    const auto lane_values = [&](int64_t l) -> const void* {
        return static_cast<const char*>(side.values) + static_cast<size_t>(l) * lane_bytes;
    };
    // Until this call settles them, no marked lane holds a factor a solve may use.
    for (int64_t l = 0; l < lanes; ++l) {
        if (!on[l]) continue;
        side.status[l] = kSingular;
        side.rescued[l] = 0;
    }
    if (raw != nullptr) {
        for (int64_t first = 0; first < lanes;) {
            if (!on[first]) {
                ++first;
                continue;
            }
            int64_t end = first + 1;
            while (end < lanes && on[end]) ++end;
            CONTRACT_TRY(AssembleLanes(reg, slot, *raw, first, end - first,
                                       static_cast<char*>(side.values) +
                                           static_cast<size_t>(first) * lane_bytes,
                                       stream));
            first = end;
        }
    }
    if (side.by_lane) {
        for (int64_t l = 0; l < lanes; ++l) {
            if (!on[l]) continue;
            std::unique_ptr<Side>& own = side.rescue[l];
            if (!own) {
                CONTRACT_TRY(CreateSide(reg, slot, ValueFill::Csr(lane_values(l)), stream, &own));
            }
            own->generation = side.generations[l];
            bool ran = false;
            bool used_refactorization = false;
            // The batch reads its lanes' statuses on the host, so each lane settles at once.
            CONTRACT_TRY(FactorSide(reg, slot, *own, ValueFill::Csr(lane_values(l)), stream,
                                    refactorization, &ran, &used_refactorization, true));
            if (ran) ++reg.stats.lanes_factored_alone;
            ++*phases;
            side.factored[l] = 1;
            side.rescued[l] = 1;
            side.status[l] = own->status;
        }
        return ffi::Error::Success();
    }
    if (side.stacked) {
        return FactorStackedSide(reg, slot, side, on, refactorization, stream, phases);
    }
    // A lane with non-finite values runs through the numeric phase like the others (cuDSS factors
    // it to non-finite entries that stay in its lane) and is reported NONFINITE from the scan.
    std::vector<uint8_t> refactored(static_cast<size_t>(lanes), 0);
    for (auto& tile_holder : side.tiles) {
        BatchTile& tile = *tile_holder;
        const int64_t count = TileCount(tile, on);
        if (count == 0) continue;
        bool refactor = refactorization;
        for (int64_t i = 0; i < tile.lanes; ++i) {
            const int64_t l = tile.first + i;
            if (on[l]) refactor = refactor && side.factored[l] != 0;
        }
        CONTRACT_TRY(TileNumericPhase(reg, slot, tile, on, count, refactor, stream));
        ++*phases;
        for (int64_t i = 0; i < tile.lanes; ++i) {
            const int64_t l = tile.first + i;
            if (!on[l]) continue;
            side.factored[l] = 1;
            refactored[l] = refactor ? 1 : 0;
        }
    }
    std::vector<LaneScan> scan;
    CONTRACT_TRY(StatusSolve(reg, slot, side, on, stream, &scan));
    std::vector<uint8_t> finite(static_cast<size_t>(lanes), 1);
    for (int64_t l = 0; l < lanes; ++l) {
        if (!on[l] || scan[l].values_finite) continue;
        finite[l] = 0;
        side.status[l] = kNonfinite;
        ++reg.stats.nonfinite;
    }
    const double bound = BatchSuspectRatio(slot.dtype) * diff_soup_status::kProbeSolutionLargest;
    const auto suspect = [&](int64_t l) {
        const LaneScan& s = scan[l];
        return on[l] != 0 && finite[l] != 0 && !(s.finite && s.largest <= bound);
    };
    // A refactorization that left a lane suspect kept a pivot sequence that no longer serves: those
    // lanes are factored afresh.
    std::vector<uint8_t> again(static_cast<size_t>(lanes), 0);
    bool any_again = false;
    for (int64_t l = 0; l < lanes; ++l) {
        if (refactored[l] && suspect(l)) {
            again[l] = 1;
            any_again = true;
        }
    }
    if (any_again) {
        for (auto& tile_holder : side.tiles) {
            BatchTile& tile = *tile_holder;
            const int64_t count = TileCount(tile, again);
            if (count == 0) continue;
            CONTRACT_TRY(TileNumericPhase(reg, slot, tile, again, count, false, stream));
            ++*phases;
        }
        std::vector<LaneScan> rescan;
        CONTRACT_TRY(StatusSolve(reg, slot, side, again, stream, &rescan));
        for (int64_t l = 0; l < lanes; ++l) {
            if (!again[l]) continue;
            scan[l].finite = rescan[l].finite;
            scan[l].largest = rescan[l].largest;
        }
    }
    for (int64_t l = 0; l < lanes; ++l) {
        if (on[l] && finite[l] && !suspect(l)) side.status[l] = kSuccess;
    }
    for (int64_t l = 0; l < lanes; ++l) {
        if (!suspect(l)) continue;
        std::unique_ptr<Side>& rescue = side.rescue[l];
        if (!rescue) {
            CONTRACT_TRY(CreateSide(reg, slot, ValueFill::Csr(lane_values(l)), stream, &rescue));
        }
        rescue->generation = side.generations[l];
        bool rescue_ran = false;
        bool used_refactorization = false;
        // The batch reads its lanes' statuses on the host, so a rescue settles its verdict at once.
        CONTRACT_TRY(FactorSide(reg, slot, *rescue, ValueFill::Csr(lane_values(l)), stream, false,
                                &rescue_ran, &used_refactorization, true));
        if (rescue_ran) ++reg.stats.lane_rescues;
        side.rescued[l] = 1;
        side.status[l] = rescue->status;
    }
    // The permanent device bytes of each tile's factor, read from the library once after its
    // first factorization (a host read of its own inside the library).
    for (auto& tile_holder : side.tiles) {
        BatchTile& tile = *tile_holder;
        if (tile.facts_known || TileCount(tile, side.factored) == 0) continue;
        int64_t estimates[6] = {-1, -1, -1, -1, -1, -1};
        QueryMemoryEstimates(slot.handle, tile.data, estimates);
        if (estimates[0] >= 0) tile.permanent_device_bytes = estimates[0];
        tile.facts_known = true;
    }
    return ffi::Error::Success();
}

// Point the solve-time matrices of a tile at the call's buffers: the tile's first lane at
// `b` + first * n * nrhs entries, and the same in `x` (cuDSS takes lane i of the tile i * n * nrhs
// entries further). The matrices are recreated when the number of right-hand sides changed;
// buffers already pointed at are left alone.
ffi::Error PointTileAt(const Slot& slot, BatchTile& tile, const void* b, void* x, int64_t nrhs) {
    const int64_t n = slot.analysis->n;
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (nrhs > limit / std::max<int64_t>(n * tile.lanes, 1)) {
        return InvalidCall("a batched solve of " + std::to_string(nrhs) +
                           " right-hand sides per lane over " + std::to_string(tile.lanes) +
                           " lanes of " + std::to_string(n) +
                           " rows holds more than 2**31 - 1 entries per batch; solve at most " +
                           std::to_string(std::max<int64_t>(limit / (n * tile.lanes), 1)) +
                           " right-hand sides per lane in one call.");
    }
    const size_t lane_bytes = static_cast<size_t>(n * nrhs) * slot.value_size;
    const char* tile_b = static_cast<const char*>(b) + static_cast<size_t>(tile.first) * lane_bytes;
    char* tile_x = static_cast<char*>(x) + static_cast<size_t>(tile.first) * lane_bytes;
    const bool recreate = tile.b == nullptr || tile.x == nullptr || tile.nrhs != nrhs;
    if (!recreate && tile.last_b == tile_b && tile.last_x == tile_x) {
        return ffi::Error::Success();
    }
    tile.last_b = nullptr;
    tile.last_x = nullptr;
    if (recreate) {
        // Cleared before the status is checked, so the destructor never repeats a destroy.
        if (tile.b != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(tile.b);
            tile.b = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy b of a batched factor");
        }
        if (tile.x != nullptr) {
            const cudssStatus_t destroyed = cudssMatrixDestroy(tile.x);
            tile.x = nullptr;
            CONTRACT_CUDSS(destroyed, "cudssMatrixDestroy x of a batched factor");
        }
        CONTRACT_CUDSS(cudssMatrixCreateDn(&tile.b, n, nrhs, n, const_cast<char*>(tile_b),
                                           slot.dtype, CUDSS_LAYOUT_COL_MAJOR),
                       "cudssMatrixCreateDn b of a batched factor");
        CONTRACT_CUDSS(cudssMatrixCreateDn(&tile.x, n, nrhs, n, tile_x, slot.dtype,
                                           CUDSS_LAYOUT_COL_MAJOR),
                       "cudssMatrixCreateDn x of a batched factor");
        tile.nrhs = nrhs;
    } else {
        CONTRACT_CUDSS(cudssMatrixSetValues(tile.b, const_cast<char*>(tile_b)),
                       "cudssMatrixSetValues b of a batched factor");
        CONTRACT_CUDSS(cudssMatrixSetValues(tile.x, tile_x),
                       "cudssMatrixSetValues x of a batched factor");
    }
    tile.last_b = tile_b;
    tile.last_x = tile_x;
    return ffi::Error::Success();
}

// Solve every lane of the batch side whose entry of `status` is SUCCESS, for `nrhs` right-hand
// sides per lane (lane i of `b` and `x` at i * n * nrhs entries), with A_bc^T when `transpose`
// (cuDSS's transpose solve mode on the same factor): each tile solves the lanes its batch factor
// serves in one masked call, a lane served by a single side of its own (a rescue, or every lane
// of a batch that runs lane by lane) solves alone on it, and every other lane's solution is nan.
ffi::Error SolveBatch(Registry& reg, Slot& slot, BatchSide& side,
                      const std::vector<int32_t>& status, const void* b, void* x, int64_t nrhs,
                      cudaStream_t stream, bool transpose) {
    const int64_t n = slot.analysis->n;
    const size_t lane_bytes = static_cast<size_t>(n * nrhs) * slot.value_size;
    if (lane_bytes == 0) return ffi::Error::Success();
    std::vector<uint8_t> on(static_cast<size_t>(side.lanes), 0);
    for (int64_t l = 0; l < side.lanes; ++l) on[l] = status[l] == kSuccess && !side.rescued[l];
    for (auto& tile_holder : side.tiles) {
        BatchTile& tile = *tile_holder;
        if (TileCount(tile, on) == 0) continue;
        if (tile.stacked) {
            CONTRACT_TRY(SolveStackedTile(reg, slot, side, tile, b, x, nrhs, stream, transpose));
            continue;
        }
        CONTRACT_TRY(PointTileAt(slot, tile, b, x, nrhs));
        if (ZeroFills(*slot.analysis, slot)) {
            CONTRACT_TRY(ZeroSolveOutput(
                static_cast<char*>(x) + static_cast<size_t>(tile.first) * lane_bytes,
                static_cast<size_t>(tile.lanes) * lane_bytes, stream, false));
            ++reg.stats.zero_fills;
        }
        CONTRACT_TRY(SetTileMask(slot, tile, &on));
        CONTRACT_TRY(SetTileSolveMode(tile, transpose ? CUDSS_SOLVE_MODE_TRANSPOSE
                                                      : CUDSS_SOLVE_MODE_NORMAL));
        cudssStatus_t solved = CUDSS_STATUS_SUCCESS;
        CONTRACT_TRY(ExecuteOnHandle(slot, stream, CUDSS_PHASE_SOLVE, tile.config, tile.data,
                                     tile.A, tile.x, tile.b, &solved));
        CONTRACT_CUDSS(solved, transpose ? "cudssExecute transposed solve of a batched factor"
                                         : "cudssExecute solve of a batched factor");
        ++reg.stats.batched_solves;
        if (transpose) ++reg.stats.transposed_solves;
    }
    for (int64_t l = 0; l < side.lanes; ++l) {
        char* lane_x = static_cast<char*>(x) + static_cast<size_t>(l) * lane_bytes;
        if (status[l] == kSuccess && side.rescued[l]) {
            const char* lane_b = static_cast<const char*>(b) + static_cast<size_t>(l) * lane_bytes;
            CONTRACT_TRY(SolveSide(reg, slot, *side.rescue[l], lane_b, lane_x, n * nrhs, stream,
                                   transpose));
            ++(side.by_lane ? reg.stats.lane_solves_alone : reg.stats.lane_rescue_solves);
        } else if (status[l] != kSuccess) {
            // Every byte 0xFF is a nan in float32 and float64.
            CONTRACT_CUDA(cudaMemsetAsync(lane_x, 0xFF, lane_bytes, stream),
                          "cudaMemsetAsync (nan solution)");
        }
    }
    return ffi::Error::Success();
}

// --------------------------------------------------------------------------
// Requests on one slot
// --------------------------------------------------------------------------

// Solve on the single slot `slot` for a handle of generation `generation`: the slot's side is
// built, or factored again from the values of `fill` when it holds another generation, and the
// `total` right-hand-side entries at `b` are solved into `x`, with A_bc^T when the call solves
// with the transpose of the handle's matrix (`transpose`) of an "A" factor, or without it of an
// "AT" factor (cuDSS's transpose solve mode on the side's factor of A_bc); a slot declared
// symmetric or symmetric positive definite solves in normal mode either way. `status` receives
// the status the handle's factor has; a status other than SUCCESS writes nan into `x`. While the
// factor's verdict is pending the status is the device's (QueueSolveVerdict): it is written to
// `device_out` when that is not null. Otherwise, with `device_out` and at least one
// right-hand-side entry, the status is written to `device_out` by the kernel that fills `x`
// before the solve (QueueFillAndWords). `status_source` receives the device word that holds the
// status once the call's work has run (`device_out` whenever this call writes it), null when
// only `status` holds it. `trusted` says the caller trusts the slot's shared storage
// (SolveStored); a solve that does not and factors a storage slot's side again, in either
// orientation, clears the slot's trust.
ffi::Error SolveSingle(Registry& reg, Slot& slot, int32_t generation, bool transpose,
                       const ValueFill& fill, const void* b, void* x, int64_t total,
                       cudaStream_t stream, int32_t* status_out, int32_t* device_out,
                       const int32_t** status_source, bool trusted) {
    const bool transposed_mode = !slot.symmetric && !slot.spd && (slot.transposed != transpose);
    CONTRACT_TRY(BindSlot(slot, stream));
    Side* side = nullptr;
    CONTRACT_TRY(EnsureSide(reg, slot, fill, stream, &side));
    if (side->generation != generation) {
        // The side holds no factor, or a factor of another request. When the side holds a
        // factor, the side first assembles the values this solve carries into its value copy,
        // comparing them with the values it holds (one read of one word): when they are equal
        // the side already holds this handle's factor and takes its generation without a
        // factorization (counted in rebuilds_avoided). Otherwise the side is factored from
        // those values (a rebuild).
        bool changed = true;
        ValueFill source = fill;
        if (side->generation != 0 && side->factored && fill.raw != nullptr) {
            ValueCheck check;
            CONTRACT_TRY(BeginValueCheck(reg, slot, 1, &check));
            CONTRACT_TRY(QueueValueCheck(reg, slot, *fill.raw, nullptr, nullptr, nullptr, fill.lane,
                                         1, side->values, check, 0, stream));
            CONTRACT_TRY(ReadValueCheck(check, stream));
            changed = check.Host()[0] != 0;
            source = ValueFill{};  // the copy holds the solve's values now
        }
        side->generation = generation;
        if (!changed) {
            ++reg.stats.rebuilds_avoided;
        } else {
            // The side of shared storage now holds this solve's matrix, not the one a refactor
            // of this execution put there, whichever orientation the solve applies.
            if (!trusted && slot.storage != 0) slot.trusted_run = -1;
            bool ran = false;
            bool used_refactorization = false;
            CONTRACT_TRY(
                FactorSide(reg, slot, *side, source, stream, false, &ran, &used_refactorization));
            if (ran) ++reg.stats.factorizations;
            ++reg.stats.rebuilds;
        }
    }

    if (!side->pending && side->retry != kRetryNone) {
        // The host has settled a failure of this factor that asks for a full factorization (or
        // a new analysis first): it runs now, from the values the factor was built from.
        bool ran = false;
        bool used_refactorization = false;
        CONTRACT_TRY(
            FactorSide(reg, slot, *side, ValueFill{}, stream, false, &ran, &used_refactorization));
        if (ran) ++reg.stats.factorizations;
    }

    *status_source = nullptr;
    if (side->pending) {
        // The factor's verdict is on the device: solve, then one kernel takes the status from
        // the device word (SINGULAR when finite right-hand sides gave a non-finite solution of
        // finite values), writes nan into `x` when the factor failed, and writes the status.
        if (total > 0) {
            CONTRACT_TRY(SolveSide(reg, slot, *side, b, x, total, stream, transposed_mode));
            ++reg.stats.solves;
        }
        CONTRACT_CUDA(diff_soup_status::QueueSolveVerdict(stream, slot.dtype == CUDA_R_64F, b, x,
                                                          total, side->verdict, device_out),
                      "solve status kernel");
        *status_source = device_out != nullptr ? device_out : &side->verdict->solve_status;
        *status_out = kSuccess;
        return MarkSlotUse(slot, stream);
    }
    const int32_t status = side->status;
    const size_t out_bytes = static_cast<size_t>(total) * slot.value_size;
    // The host knows the status before anything is queued, so a one-lane call has it written into
    // its status output by the kernel that fills the solution (zeroes before the solve, or nan),
    // not by a launch of its own.
    diff_soup_assembly::ResultWords words;
    const bool fused = device_out != nullptr && out_bytes > 0;
    if (fused) {
        words.count = 1;
        words.targets[0] = device_out;
        words.values[0] = status;
    }
    if (status != kSuccess) {
        if (fused) {
            // Every byte 0xFF is a nan in float32 and float64.
            CONTRACT_CUDA(diff_soup_assembly::QueueFillAndWords(stream, x, out_bytes, 0xFF, words),
                          "nan solution and result words kernel");
        } else if (out_bytes > 0) {
            CONTRACT_CUDA(cudaMemsetAsync(x, 0xFF, out_bytes, stream),
                          "cudaMemsetAsync (nan solution)");
        }
    } else if (total > 0) {
        CONTRACT_TRY(SolveSide(reg, slot, *side, b, x, total, stream, transposed_mode,
                               fused ? &words : nullptr));
        ++reg.stats.solves;
    }
    if (fused) *status_source = device_out;
    CONTRACT_TRY(MarkSlotUse(slot, stream));
    *status_out = status;
    return ffi::Error::Success();
}

// Solve on the batch slot `slot` for a handle whose lane l is at generation `generations[l]`,
// every lane at once, with the transpose of the handle's matrix when `transpose`. The batch side
// factors A_bc whatever the slot's orientation, and solves with A_bc^T in cuDSS's transpose solve
// mode for a transposed solve of an "A" factor or any solve of an "AT" factor (normal mode either
// way on a slot declared symmetric or symmetric positive definite). The batch side is built on
// first use. A lane whose factor is of another generation than its handle's (a stale lane: the
// side was never factored for it, was invalidated, or holds a later factor of that lane, for
// instance one a vmapped loop computed for a lane that had already left the loop) has its lane of
// `raw` (the call's values, the values the handle was built from) assembled into the side's value
// copy. When every stale lane's factor stands for some request, the assembly compares the values
// with the copy it replaces (one read of one word per lane): a stale lane whose copy held these
// very values already holds its handle's factor and takes its generation without a factorization
// (a tile, or a lane of a batch that runs lane by lane, with no other stale lane is counted in
// rebuilds_avoided). The other stale lanes are factored (alone: a masked factorization, counted
// as rebuilds) before the solve; those lanes then stand for their handle's generation. The `nrhs`
// right-hand sides per lane at `b` are solved into `x`. A lane marked in `empty` (when given)
// holds no factor: it takes nan and the status EMPTY. `statuses` receives each lane's status.
// `trusted` says the caller trusts the slot's shared storage (SolveStored); a solve that does not
// and factors lanes of a storage slot again clears the slot's trust.
ffi::Error SolveBatchSlot(Registry& reg, Slot& slot, const int32_t* generations, bool transpose,
                          const std::vector<uint8_t>* empty, const RawValues& raw, const void* b,
                          void* x, int64_t nrhs, cudaStream_t stream, int32_t* statuses,
                          bool trusted) {
    const bool transposed_mode = !slot.symmetric && !slot.spd && (slot.transposed != transpose);
    CONTRACT_TRY(BindSlot(slot, stream));
    const int64_t lanes = slot.lanes;
    const auto is_empty = [&](int64_t l) { return empty != nullptr && (*empty)[l] != 0; };
    BatchSide* side = nullptr;
    CONTRACT_TRY(EnsureBatchSide(reg, slot, ValueFill::Raw(raw), stream, &side));
    std::vector<uint8_t> stale(static_cast<size_t>(lanes), 0);
    bool any_stale = false;
    for (int64_t l = 0; l < lanes; ++l) {
        if (is_empty(l) || side->generations[l] == generations[l]) continue;
        stale[l] = 1;
        any_stale = true;
    }
    if (any_stale) {
        const size_t lane_bytes = static_cast<size_t>(slot.analysis->nse) * slot.value_size;
        bool check_values = true;
        for (int64_t l = 0; l < lanes; ++l) {
            if (stale[l] && (side->generations[l] == 0 || !side->factored[l])) {
                check_values = false;
            }
        }
        ValueCheck check;
        if (check_values) CONTRACT_TRY(BeginValueCheck(reg, slot, lanes, &check));
        for (int64_t first = 0; first < lanes;) {
            if (!stale[first]) {
                ++first;
                continue;
            }
            int64_t end = first + 1;
            while (end < lanes && stale[end]) ++end;
            void* target =
                static_cast<char*>(side->values) + static_cast<size_t>(first) * lane_bytes;
            if (check_values) {
                CONTRACT_TRY(QueueValueCheck(reg, slot, raw, nullptr, nullptr, nullptr, first,
                                             end - first, target, check, first, stream));
            } else {
                CONTRACT_TRY(AssembleLanes(reg, slot, raw, first, end - first, target, stream));
            }
            first = end;
        }
        std::vector<uint8_t> changed = stale;
        if (check_values) {
            CONTRACT_TRY(ReadValueCheck(check, stream));
            for (int64_t l = 0; l < lanes; ++l) {
                changed[l] = stale[l] && check.Host()[l] != 0 ? 1 : 0;
            }
        }
        bool any_changed = false;
        for (int64_t l = 0; l < lanes; ++l) {
            if (stale[l]) side->generations[l] = generations[l];
            any_changed = any_changed || changed[l] != 0;
        }
        if (side->by_lane) {
            for (int64_t l = 0; l < lanes; ++l) {
                if (stale[l] && !changed[l]) ++reg.stats.rebuilds_avoided;
            }
        } else {
            for (const auto& tile : side->tiles) {
                if (TileCount(*tile, stale) > 0 && TileCount(*tile, changed) == 0) {
                    ++reg.stats.rebuilds_avoided;
                }
            }
        }
        if (any_changed) {
            // The shared storage now holds this solve's matrices in the changed lanes, not the
            // ones a refactor of this execution put there.
            if (!trusted && slot.storage != 0) slot.trusted_run = -1;
            int64_t phases = 0;
            CONTRACT_TRY(
                FactorBatchSide(reg, slot, *side, nullptr, stream, false, &changed, &phases));
            reg.stats.rebuilds += phases;
        }
    }
    std::vector<int32_t> lane_status(static_cast<size_t>(lanes), kEmpty);
    for (int64_t l = 0; l < lanes; ++l) {
        if (!is_empty(l)) lane_status[l] = side->status[l];
    }
    CONTRACT_TRY(SolveBatch(reg, slot, *side, lane_status, b, x, nrhs, stream, transposed_mode));
    CONTRACT_TRY(MarkSlotUse(slot, stream));
    std::copy(lane_status.begin(), lane_status.end(), statuses);
    return ffi::Error::Success();
}

// The values of a refactor that returns the values its result holds, in the caller's layout:
// `raw` the call's values, `held` the held factor's values and `kept` the result (the same
// buffer as `raw` when XLA honours the alias), `lane_bytes` bytes per lane each.
struct KeptValues {
    const char* raw = nullptr;
    const char* held = nullptr;
    char* kept = nullptr;
    size_t lane_bytes = 0;
};

// Copy into `kept` each lane's values: the held factor's for a lane that does not ask (`asks`,
// one byte per lane), the call's for a lane that asks (nothing moves when the result is the
// call's buffer). Runs of neighbouring lanes with the same source are copied at once.
ffi::Error CopyKeptValues(const KeptValues& kept, const uint8_t* asks, int64_t lanes,
                          cudaStream_t stream) {
    if (kept.lane_bytes == 0) return ffi::Error::Success();
    int64_t first = 0;
    while (first < lanes) {
        const bool ask = asks[first] != 0;
        int64_t end = first + 1;
        while (end < lanes && (asks[end] != 0) == ask) ++end;
        const size_t offset = static_cast<size_t>(first) * kept.lane_bytes;
        const char* source = (ask ? kept.raw : kept.held) + offset;
        char* target = kept.kept + offset;
        if (source != target) {
            CONTRACT_CUDA(cudaMemcpyAsync(target, source,
                                          static_cast<size_t>(end - first) * kept.lane_bytes,
                                          cudaMemcpyDeviceToDevice, stream),
                          "cudaMemcpyAsync (kept values)");
        }
        first = end;
    }
    return ffi::Error::Success();
}

// One refactor request on the single slot `held` at `site`; a null `held` is an empty handle
// (slot id `empty_id`, negative), whose slot at the site is keyed by `empty_key`. With
// `refactor` false the held handle comes back unchanged (an empty one with the status EMPTY),
// and a site that does not hold the slot records that its call for that held slot returned it.
// Otherwise the values of `fill` are factored into the held slot (in place, when the site holds
// it or `adopt` is set) or into the site's slot for it. `out` receives the handle and status:
// context id, slot id, generation, status; `status_source` the device word that holds the
// status instead, while the factor's verdict is pending (null otherwise). The call has read its
// predicate on the host, so it settles a pending verdict of the factor it keeps or factors anew
// (one wait) and runs the full factorization (or new analysis) a settled failure asks for: a
// held factor it keeps is factored again from the values it was built from.
ffi::Error RefactorSingle(Registry& reg, Site& site, Slot* held, int32_t empty_id,
                          int64_t empty_key, int32_t context, int32_t held_generation,
                          bool refactor, bool adopt, const ValueFill& fill, const SlotSpec& spec,
                          int device, cudaStream_t stream, int32_t* out,
                          const int32_t** status_source, Slot* target = nullptr) {
    *status_source = nullptr;
    if (!refactor) {
        out[0] = context;
        if (held == nullptr) {
            out[1] = empty_id;
            out[2] = held_generation;
            out[3] = kEmpty;
            return ffi::Error::Success();
        }
        if (held->site != site.id) site.forward_of_key[SiteKey{context, held->id}] = held->id;
        out[1] = held->id;
        out[2] = held_generation;
        if (!held->primary) {
            out[3] = kSingular;
            return ffi::Error::Success();
        }
        Side& kept = *held->primary;
        if (kept.pending || kept.retry != kRetryNone) {
            CONTRACT_TRY(CheckDevice(*held, device));
            CONTRACT_TRY(BindSlot(*held, stream));
            CONTRACT_TRY(SettleSide(reg, *held, kept, stream));
            if (kept.retry != kRetryNone) {
                bool ran = false;
                bool used_refactorization = false;
                CONTRACT_TRY(FactorSide(reg, *held, kept, ValueFill{}, stream, false, &ran,
                                        &used_refactorization));
                if (ran) ++reg.stats.factorizations;
            }
            CONTRACT_TRY(MarkSlotUse(*held, stream));
        }
        out[3] = SideStatus(kept, status_source);
        return ffi::Error::Success();
    }
    if (held != nullptr) site.forward_of_key.erase(SiteKey{context, held->id});
    // In place when the held factor is already this site's, or when the call adopts the held
    // slot whichever site owns it (the site then forwards that slot, as it does for a call that
    // returned it unchanged, so static paths through the site resolve to it); otherwise the
    // site's slot for that held factor, so other holders of it keep it.
    const bool in_place = held != nullptr && (held->site == site.id || adopt);
    Slot* slot = held;
    if (in_place) {
        const bool adopted = held->site != site.id;
        CONTRACT_TRY(CheckInPlace(*held, spec, adopted));
        if (adopted) site.forward_of_key[SiteKey{context, held->id}] = held->id;
    } else {
        if (held == nullptr) CONTRACT_TRY(CheckContext(reg, context));
        const int64_t key = held != nullptr ? held->id : empty_key;
        if (target != nullptr && held == nullptr) {
            slot = target;
        } else {
            CONTRACT_TRY(SiteSlot(reg, site, context, key, spec, device, false, 1, &slot));
        }
        CONTRACT_TRY(CheckDevice(*slot, device));
    }
    CONTRACT_TRY(BindSlot(*slot, stream));
    Side* side = nullptr;
    CONTRACT_TRY(EnsureSide(reg, *slot, fill, stream, &side));
    const int32_t generation = NextGeneration(*slot);
    side->generation = generation;
    bool ran = false;
    bool used_refactorization = false;
    // A refactorization's verdict stays pending; when it ends singular (its kept pivot order no
    // longer serves), the next call that settles it factors the side afresh.
    CONTRACT_TRY(FactorSide(reg, *slot, *side, fill, stream, in_place, &ran,
                            &used_refactorization));
    if (ran && used_refactorization) {
        ++reg.stats.refactorizations;
    } else if (ran) {
        ++reg.stats.factorizations;
    }
    CONTRACT_TRY(MarkSlotUse(*slot, stream));
    out[0] = context;
    out[1] = slot->id;
    out[2] = generation;
    out[3] = SideStatus(*side, status_source);
    return ffi::Error::Success();
}

// One refactor request on a batch of `lanes` lanes at `site`, with one predicate per lane
// (`asks`). `held` is the batch slot the handle names, null when every lane is empty; lane l of
// the handle is (`contexts[l]`, `ids[l]`, `generations[l]`) (`contexts` and `ids` may be null
// when no lane is empty: `context` and the held slot's lane ids); `empty` (when given) marks
// the empty lanes (a negative slot id).
//
// With no lane asking the handles come back unchanged and nothing is factored (a site that
// does not hold the slot records that it returned it). With any lane asking, the held slot (in
// place, by refactorization, when the site holds it or `adopt` is set) or the site's batch slot
// for it (for a handle whose every lane is empty, the site's batch slot for the context and -1)
// factors the lanes that need it, one masked numeric phase per tile (FactorBatchSide):
//   * a lane that asks is factored from its lane of `raw` (the call's values, assembled into
//     the slot's value copy) and takes the request's new generation;
//   * a lane that does not ask keeps its held factor: in place it is not factored and keeps its
//     value copy, its factor, its status and its generation; into the site's slot for another
//     held slot it takes the held slot's value copy, is factored and takes the new generation. A
//     lane whose held slot no longer holds the factor its handle names (its lane is at another
//     generation) is stale: it is not factored and comes back with its handle's generation (the
//     new one in the site's slot), its held status and a side lane that stands for no handle, so
//     the next solve factors it again from the values that solve carries (the held values the
//     call carries are not the factor's);
//   * an empty lane that does not ask stays empty: its handle comes back unchanged with the
//     status EMPTY, its slot lane is left as it is, and it is never rescued.
// `words` receives the handle and status arrays: context ids, slot ids, generations, statuses,
// `lanes` entries each.
ffi::Error RefactorBatchSlot(Registry& reg, Site& site, Slot* held,
                             const std::vector<uint8_t>* empty, int32_t context,
                             const int32_t* contexts, const int32_t* ids,
                             const int32_t* generations, const uint8_t* asks,
                             const RawValues& raw, int64_t lanes, bool adopt,
                             const SlotSpec& spec, int device, cudaStream_t stream,
                             int32_t* words, Slot* target = nullptr) {
    const auto is_empty = [&](int64_t l) { return empty != nullptr && (*empty)[l] != 0; };
    const auto write = [&](int64_t l, int32_t slot_id, int32_t generation, int32_t status) {
        words[l] = contexts != nullptr ? contexts[l] : context;
        words[lanes + l] = slot_id;
        words[2 * lanes + l] = generation;
        words[3 * lanes + l] = status;
    };
    const auto write_empty = [&](int64_t l) {
        write(l, ids != nullptr ? ids[l] : kEmptySlot, generations[l], kEmpty);
    };
    bool any = false;
    bool all = true;
    for (int64_t l = 0; l < lanes; ++l) {
        any = any || asks[l] != 0;
        all = all && asks[l] != 0;
    }
    if (!any) {
        if (held != nullptr && held->site != site.id) {
            site.forward_of_key[SiteKey{context, held->id}] = held->id;
        }
        for (int64_t l = 0; l < lanes; ++l) {
            if (is_empty(l)) {
                write_empty(l);
                continue;
            }
            write(l, static_cast<int32_t>(held->id + l), generations[l],
                  held->batch_primary ? held->batch_primary->status[l] : kSingular);
        }
        return ffi::Error::Success();
    }
    if (held != nullptr) site.forward_of_key.erase(SiteKey{context, held->id});
    const bool in_place = held != nullptr && (held->site == site.id || adopt);
    Slot* slot = held;
    if (in_place) {
        const bool adopted = held->site != site.id;
        CONTRACT_TRY(CheckInPlace(*held, spec, adopted));
        if (adopted) site.forward_of_key[SiteKey{context, held->id}] = held->id;
    } else {
        if (held == nullptr) CONTRACT_TRY(CheckContext(reg, context));
        const int64_t key = held != nullptr ? held->id : kEmptySlot;
        if (target != nullptr && held == nullptr) {
            slot = target;
        } else {
            CONTRACT_TRY(SiteSlot(reg, site, context, key, spec, device, true, lanes, &slot));
        }
        CONTRACT_TRY(CheckDevice(*slot, device));
    }
    CONTRACT_TRY(BindSlot(*slot, stream));
    const bool fresh = !slot->batch_primary;
    BatchSide* side = nullptr;
    CONTRACT_TRY(EnsureBatchSide(reg, *slot, ValueFill::Raw(raw), stream, &side));
    const BatchSide* held_side = held != nullptr ? held->batch_primary.get() : nullptr;
    // Statuses of the held factor, taken before an in-place factorization overwrites them.
    const std::vector<int32_t> held_status =
        held_side != nullptr ? held_side->status
                             : std::vector<int32_t>(static_cast<size_t>(lanes), kSingular);
    // Every batch side holds the values of A_bc, whatever its slot's orientation.
    const bool same_layout = held != nullptr && held->dtype == slot->dtype;
    const auto held_current = [&](int64_t l) {
        return held_side != nullptr && same_layout && held_side->generations[l] == generations[l];
    };
    const int32_t generation = NextGeneration(*slot);
    // Per lane: where its value copy comes from (0 the call's values, 1 the held slot's value
    // copy, 2 nothing to copy), whether it is stale, and the generation its handle comes back
    // with.
    std::vector<int> source(static_cast<size_t>(lanes), 0);
    std::vector<uint8_t> stale(static_cast<size_t>(lanes), 0);
    std::vector<int32_t> out_generation(static_cast<size_t>(lanes), generation);
    for (int64_t l = 0; l < lanes; ++l) {
        if (asks[l] != 0) continue;
        if (is_empty(l)) {
            source[l] = 2;
            out_generation[l] = generations[l];
        } else if (held_current(l)) {
            source[l] = in_place ? 2 : 1;
            if (in_place) out_generation[l] = generations[l];
        } else {
            source[l] = 2;
            stale[l] = 1;
            if (in_place) out_generation[l] = generations[l];
        }
    }
    // The lanes this request factors: those that ask, and those that take the held slot's
    // values into another slot.
    std::vector<uint8_t> factor_lanes(static_cast<size_t>(lanes), 0);
    for (int64_t l = 0; l < lanes; ++l) factor_lanes[l] = asks[l] != 0 || source[l] == 1;
    // A side created by this call holds the call's values in every lane already.
    for (int64_t l = 0; l < lanes; ++l) {
        if (fresh && source[l] == 0) source[l] = 2;
    }
    // The value copy, a run of neighbouring lanes with one source at a time: the call's values
    // assembled in, or the held slot's value copy (already in CSR order) copied over.
    {
        const size_t lane_bytes = static_cast<size_t>(slot->analysis->nse) * slot->value_size;
        for (int64_t first = 0; first < lanes;) {
            const int from = source[first];
            int64_t end = first + 1;
            while (end < lanes && source[end] == from) ++end;
            const size_t offset = static_cast<size_t>(first) * lane_bytes;
            char* target = static_cast<char*>(side->values) + offset;
            if (from == 0) {
                CONTRACT_TRY(AssembleLanes(reg, *slot, raw, first, end - first, target, stream));
            } else if (from == 1) {
                CONTRACT_CUDA(cudaMemcpyAsync(target,
                                              static_cast<const char*>(held_side->values) + offset,
                                              static_cast<size_t>(end - first) * lane_bytes,
                                              cudaMemcpyDeviceToDevice, stream),
                              "cudaMemcpyAsync (batched factor values of some lanes)");
            }
            first = end;
        }
    }
    for (int64_t l = 0; l < lanes; ++l) {
        if (is_empty(l) && asks[l] == 0) continue;  // its slot lane is left as it is
        side->generations[l] = stale[l] ? 0 : out_generation[l];
    }
    int64_t phases = 0;
    CONTRACT_TRY(FactorBatchSide(reg, *slot, *side, nullptr, stream, in_place, &factor_lanes,
                                 &phases));
    slot->lane_generations.assign(static_cast<size_t>(lanes), kEmptySlot);
    for (int64_t l = 0; l < lanes; ++l) {
        if (is_empty(l) && asks[l] == 0) {
            side->status[l] = kEmpty;
            write_empty(l);
            continue;
        }
        slot->lane_generations[l] = out_generation[l];
        write(l, static_cast<int32_t>(slot->id + l), out_generation[l],
              stale[l] ? held_status[l] : side->status[l]);
    }
    CONTRACT_TRY(MarkSlotUse(*slot, stream));
    return ffi::Error::Success();
}

// A solve with a factor that names shared storage (the slot `slot`, single or batched): the
// storage is trusted to hold the factor's matrix, as the refactor of such a factor makes it, so
// the primary is used as it stands, whatever generation the handle carries. Only a side (a
// lane) that stands for no factor (an invalidated one) is factored first, from the values the
// solve carries (a rebuild). `lane_total` right-hand-side entries per lane at `b` are solved into
// `x`; `statuses` receives each lane's status (a single slot's from the device as SolveSingle
// says, with `device_out` and `status_source`).
ffi::Error SolveStored(Registry& reg, Slot& slot, const RawValues& raw, const void* b, void* x,
                       int64_t lane_total, cudaStream_t stream, int32_t* statuses,
                       int32_t* device_out, const int32_t** status_source) {
    if (!slot.batch) {
        int32_t generation = slot.primary->generation;
        if (generation == 0) generation = NextGeneration(slot);
        return SolveSingle(reg, slot, generation, false, ValueFill::Raw(raw), b, x, lane_total,
                           stream, &statuses[0], device_out, status_source, true);
    }
    std::vector<int32_t> generations = slot.batch_primary->generations;
    int32_t fresh = 0;
    for (int32_t& g : generations) {
        if (g != 0) continue;
        if (fresh == 0) fresh = NextGeneration(slot);
        g = fresh;
    }
    return SolveBatchSlot(reg, slot, generations.data(), false, nullptr, raw, b, x,
                          lane_total / slot.analysis->n, stream, statuses, true);
}

// Refactor a single side in place from the values its copy holds, by cuDSS's refactorization (a
// full factorization when the side holds none, or when its settled verdict asks for one: a
// refactorization that ended singular is factored afresh by the next call that settles it),
// counting what ran. `result` as FactorSide.
ffi::Error RefactorSideInPlace(Registry& reg, Slot& slot, Side& side, cudaStream_t stream,
                               const diff_soup_status::VerdictResult* result = nullptr) {
    bool ran = false;
    bool used_refactorization = false;
    CONTRACT_TRY(FactorSide(reg, slot, side, ValueFill{}, stream, true, &ran,
                            &used_refactorization, false, result));
    if (ran && used_refactorization) {
        ++reg.stats.refactorizations;
    } else if (ran) {
        ++reg.stats.factorizations;
    }
    return ffi::Error::Success();
}

// A refactor of a factor that names shared storage, on the filled storage slot `slot` (single
// or batched, `lanes` lanes) in `context`: the storage is made to hold the matrix the result
// stands for. One kernel assembles into the primary's value copy, in place, the values each lane
// keeps (the call's where its predicate holds, the held factor's elsewhere) while comparing them
// with the copy, and one read brings back a word per lane (QueueAssemblyCompare): a lane whose
// predicate holds or whose values changed (for instance the first attempt of a replayed segment,
// whose restored values the storage no longer holds) is factored, in place; every other lane
// keeps its factor and its generation; a lane that does not ask and whose handle is empty stays
// empty and leaves the storage alone. A side (a lane) that stands for no factor (invalidated) is
// factored too. `asks` receives one byte per lane (its predicate) and `words` the handle and
// status arrays (an empty lane comes back as the empty handle (context, -1, -1), EMPTY); on a
// single slot whose verdict is pending, `status_source` receives the device word that holds the
// status (null otherwise).
//
// The kernel of the check also writes each lane's kept values into `kept` (the caller's layout):
// the call's values where the predicate holds, the held values elsewhere. On a single slot with
// `targets` (the call's four result arrays: context ids, slot ids, generations, statuses) it also
// writes the result the call has when the storage keeps its factor: the storage's handle and the
// factor's status as they stand when the kernel runs; a refactorization of the slot has its
// result written by the verdict kernel of its numeric phase instead (QueueValuesVerdict).
// `written` says whether one of those results stands (no numeric phase ran, the lane is not
// empty and a verdict settled by the call did not change the status; or the refactorization's
// verdict is pending), in which case the caller writes no result words of its own.
ffi::Error RefactorStored(Registry& reg, Slot& slot, int32_t context, const RawValues& raw,
                          const RawValues& held_raw, const uint8_t* predicate,
                          const int32_t* held_slot, int64_t lanes, cudaStream_t stream,
                          uint8_t* asks, int32_t* words, const int32_t** status_source,
                          void* kept, int32_t* const* targets, bool* written) {
    *status_source = nullptr;
    *written = false;
    using diff_soup_assembly::kLaneAsks;
    using diff_soup_assembly::kLaneChanged;
    using diff_soup_assembly::kLaneEmpty;
    const auto write = [&](int64_t l, int32_t slot_id, int32_t generation, int32_t status) {
        words[l] = context;
        words[lanes + l] = slot_id;
        words[2 * lanes + l] = generation;
        words[3 * lanes + l] = status;
    };
    CONTRACT_TRY(BindSlot(slot, stream));
    ValueCheck check;
    CONTRACT_TRY(BeginValueCheck(reg, slot, lanes, &check));
    void* copy = slot.batch ? slot.batch_primary->values : slot.primary->values;
    // The result of a single slot that keeps its factor, written by the check's kernel.
    diff_soup_assembly::ResultWords early;
    const bool single_result = !slot.batch && targets != nullptr && slot.analysis->nse > 0;
    const bool pending_at_launch = !slot.batch && slot.primary->pending;
    if (single_result) {
        const Side& side = *slot.primary;
        early.count = 4;
        for (int k = 0; k < 4; ++k) early.targets[k] = targets[k];
        early.values[0] = context;
        early.values[1] = slot.id;
        early.values[2] = side.generation;
        early.values[3] = SideStatus(side, &early.sources[3]);
    }
    const int32_t early_generation = single_result ? slot.primary->generation : 0;
    CONTRACT_TRY(QueueValueCheck(reg, slot, raw, &held_raw, predicate, held_slot, 0,
                                 lanes, copy, check, 0, stream, kept,
                                 single_result ? &early : nullptr));
    // The one read of the call: the check words, and on a single slot the pending verdict of the
    // primary's latest numeric phase, settled in the same wait (the check rewrites the value copy
    // only, never the factor or its status words).
    CONTRACT_TRY(QueueValueCheckRead(check, stream));
    int32_t read_word = kSuccess;
    if (pending_at_launch) {
        CONTRACT_TRY(SettleSide(reg, slot, *slot.primary, stream, true, &read_word));
    } else {
        CONTRACT_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize (value check words)");
    }
    DecodeValueCheck(check);
    ++reg.stats.host_reads;
    ++reg.stats.predicate_reads;
    const int32_t* flags = check.Host();
    for (int64_t l = 0; l < lanes; ++l) {
        asks[l] = (flags[l] & kLaneAsks) != 0 ? 1 : 0;
        if (flags[l] == kLaneChanged) ++reg.stats.storage_value_changes;
        if (flags[l] == kLaneAsks) ++reg.stats.storage_asks_unchanged;
    }
    if (!slot.batch) {
        Side& side = *slot.primary;
        if ((flags[0] & kLaneEmpty) != 0) {
            write(0, kEmptySlot, kEmptySlot, kEmpty);
        } else {
            // A lane whose values changed is factored in place; so is one whose settled verdict
            // asks for a full factorization (its values are the ones it keeps).
            const bool changed = (flags[0] & kLaneChanged) != 0 || side.generation == 0;
            if (changed) side.generation = NextGeneration(slot);
            const bool factors = changed || side.retry != kRetryNone;
            // A refactorization whose verdict stays pending has its result written by its
            // verdict kernel (the status taken from the verdict it sets).
            diff_soup_status::VerdictResult late;
            if (factors && targets != nullptr) {
                late.count = 4;
                for (int k = 0; k < 4; ++k) late.targets[k] = targets[k];
                late.values[0] = context;
                late.values[1] = slot.id;
                late.values[2] = side.generation;
                late.status_index = 3;
                CONTRACT_TRY(RefactorSideInPlace(reg, slot, side, stream, &late));
            } else if (factors) {
                CONTRACT_TRY(RefactorSideInPlace(reg, slot, side, stream));
            }
            write(0, slot.id, side.generation, SideStatus(side, status_source));
            // The check kernel's result stands when nothing ran and the status it took (the
            // settled status, or the device word the settle read) is the status now; the verdict
            // kernel's when the refactorization's verdict is pending.
            const bool early_stands = single_result && !factors &&
                                      side.generation == early_generation && !side.pending &&
                                      (!pending_at_launch || side.status == read_word);
            const bool late_stands = factors && targets != nullptr && side.pending &&
                                     side.pending_refactorization;
            *written = early_stands || late_stands;
        }
    } else {
        BatchSide& side = *slot.batch_primary;
        std::vector<uint8_t> factor(static_cast<size_t>(lanes), 0);
        bool any = false;
        for (int64_t l = 0; l < lanes; ++l) {
            if ((flags[l] & kLaneEmpty) != 0) continue;
            factor[l] = (flags[l] & kLaneChanged) != 0 || side.generations[l] == 0 ? 1 : 0;
            any = any || factor[l] != 0;
        }
        if (any) {
            // Only the lanes that are factored anew run a numeric phase (a masked
            // refactorization); every other lane keeps its factor and its generation.
            const int32_t generation = NextGeneration(slot);
            for (int64_t l = 0; l < lanes; ++l) {
                if (factor[l]) side.generations[l] = generation;
            }
            int64_t phases = 0;
            CONTRACT_TRY(
                FactorBatchSide(reg, slot, side, nullptr, stream, true, &factor, &phases));
        }
        slot.lane_generations.assign(static_cast<size_t>(lanes), kEmptySlot);
        for (int64_t l = 0; l < lanes; ++l) {
            if ((flags[l] & kLaneEmpty) != 0) {
                write(l, kEmptySlot, kEmptySlot, kEmpty);
                continue;
            }
            slot.lane_generations[l] = side.generations[l];
            write(l, static_cast<int32_t>(slot.id + l), side.generations[l], side.status[l]);
        }
    }
    return MarkSlotUse(slot, stream);
}

// --------------------------------------------------------------------------
// XLA FFI handlers
// --------------------------------------------------------------------------

template <ffi::DataType T>
SlotSpec SpecFor(bool transposed, bool symmetric, bool spd) {
    SlotSpec spec;
    spec.transposed = transposed;
    spec.symmetric = symmetric;
    spec.spd = spd;
    spec.dtype = get_cuda_data_type<T>();
    spec.value_size = sizeof(typename get_native_data_type<T>::type);
    return spec;
}

// The values operand of a call: float32 or float64 entries in the caller's coordinate order,
// (nnz,) for one lane or (lanes, nnz) for a batch. `lanes` and `batched` receive the lane count
// and whether the operand carries a lane axis.
ffi::Error RawOperand(const ffi::AnyBuffer& values, const char* what, RawValues* raw,
                      int64_t* lanes, bool* batched) {
    const auto dims = values.dimensions();
    if (dims.size() != 1 && dims.size() != 2) {
        return InvalidCall(std::string("the values of ") + what +
                           " must have shape (nnz,) or (lanes, nnz); got rank " +
                           std::to_string(dims.size()) + ".");
    }
    const ffi::DataType type = values.element_type();
    if (type == ffi::F64) {
        raw->f64 = true;
    } else if (type == ffi::F32) {
        raw->f64 = false;
    } else {
        return InvalidCall(std::string("the values of ") + what +
                           " must be float32 or float64; cast the values before the call.");
    }
    *batched = dims.size() == 2;
    *lanes = *batched ? dims[0] : 1;
    raw->nnz = *batched ? dims[1] : dims[0];
    raw->data = values.untyped_data();
    return ffi::Error::Success();
}

// The handle words of a call whose program states that its handle is the result of the call at
// the last site of its static path, in this very execution (`run_id`): the words that call wrote
// (context ids, slot ids, generations, `lanes` of each), or null. The execution's identity comes
// from XLA, so the words of another execution of the same program are never taken.
const int32_t* RunLocalWords(Registry& reg, const CallAttrs& at, int64_t run_id, int64_t lanes) {
    if (!at.RunLocalPath()) return nullptr;
    const int64_t id = at.static_path.back();
    if (id <= 0) return nullptr;
    Site* site = FindSite(reg, id);
    if (site == nullptr || site->result_run_id != run_id || site->result_lanes != lanes) {
        return nullptr;
    }
    return site->result_words.data();
}

// Keep the handle words a call at `site` wrote (`words`: context ids, slot ids and generations,
// `lanes` of each, in that order) as the site's result in execution `run_id`.
void RecordSiteResult(Site& site, int64_t run_id, int64_t lanes, const int32_t* words) {
    site.result_run_id = run_id;
    site.result_lanes = lanes;
    site.result_words.assign(words, words + 3 * lanes);
}

// Factor: values (nnz,) build or refresh a single slot; values (lanes, nnz) a batch slot. The
// values are assembled on the device into the slot's value copy. The slot's key is the static
// context id and lane token when the call passes them, the operands read on the host otherwise.
//
// Every handler takes the factor's order word as its last buffer operand and never reads it:
// the word makes the call a data dependency of the operation that produced it. The call's status
// result becomes the next order word (an invalidation, which has no other result, returns the
// order word itself, aliased to the operand and untouched).
template <ffi::DataType T>
ffi::Error FactorImpl(cudaStream_t stream, ffi::RunId run, ffi::Token, ffi::AnyBuffer values,
                      ffi::Buffer<ffi::S32> lane_token, ffi::Buffer<ffi::S32> context_id,
                      ffi::Buffer<ffi::S32> /*order*/,
                      ffi::Result<ffi::Token>, ffi::ResultBuffer<ffi::S32> out_context,
                      ffi::ResultBuffer<ffi::S32> out_slot,
                      ffi::ResultBuffer<ffi::S32> out_generation,
                      ffi::ResultBuffer<ffi::S32> out_status,
                      ffi::Dictionary attrs) {
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    CallAttrs at;
    CONTRACT_TRY(DecodeAttrs(attrs, CallKind::kFactor, &at));
    RawValues raw;
    int64_t lanes = 0;
    bool batched = false;
    CONTRACT_TRY(RawOperand(values, "a factor", &raw, &lanes, &batched));
    if (lane_token.element_count() != 1 || context_id.element_count() != 1) {
        return InvalidCall("the lane token and the context id of a factor must be int32 "
                           "scalars.");
    }
    const int64_t results[4] = {static_cast<int64_t>(out_context->element_count()),
                                static_cast<int64_t>(out_slot->element_count()),
                                static_cast<int64_t>(out_generation->element_count()),
                                static_cast<int64_t>(out_status->element_count())};
    for (int64_t count : results) {
        if (count != lanes) {
            return InvalidCall("a factor of " + std::to_string(lanes) +
                               " lanes returns handle and status leaves of " +
                               std::to_string(lanes) + " entries; got " + std::to_string(count) +
                               ".");
        }
    }
    if (lanes == 0) return ffi::Error::Success();
    int device = 0;
    CONTRACT_CUDA(cudaGetDevice(&device), "cudaGetDevice");
    const bool static_key = at.StaticKey();
    const size_t first = static_key ? 0 : 2;
    Staging* block = nullptr;
    CONTRACT_TRY(AcquireStaging(reg, device, first + 4 * static_cast<size_t>(lanes), &block));
    int64_t token = 0;
    int32_t context = -1;
    if (static_key) {
        token = at.static_lane;
        context = static_cast<int32_t>(at.static_context);
        ++reg.stats.static_dispatches;
    } else {
        const ReadSpan spans[2] = {{lane_token.typed_data(), sizeof(int32_t), 0},
                                   {context_id.typed_data(), sizeof(int32_t), 1}};
        CONTRACT_TRY(ReadSpans(stream, *block, spans, 2));
        ++reg.stats.host_reads;
        token = block->host[0];
        context = block->host[1];
    }

    CONTRACT_TRY(CheckContext(reg, context));
    Site* site = nullptr;
    CONTRACT_TRY(SiteFor(reg, at.site, at.analysis_id, &site));
    CONTRACT_TRY(CheckRaw(*site->analysis, raw));
    Slot* slot = nullptr;
    CONTRACT_TRY(SiteSlot(reg, *site, context, token,
                          SpecFor<T>(at.transposed, at.symmetric, at.spd), device, batched,
                          lanes, &slot));
    CONTRACT_TRY(CheckDevice(*slot, device));
    CONTRACT_TRY(BindSlot(*slot, stream));
    int32_t* words = block->host + first;
    const int32_t* status_source = nullptr;  // the device word of a pending verdict
    const ValueFill fill = ValueFill::Raw(raw);
    if (!batched) {
        Side* side = nullptr;
        CONTRACT_TRY(EnsureSide(reg, *slot, fill, stream, &side));
        const int32_t generation = NextGeneration(*slot);
        bool ran = false;
        bool used_refactorization = false;
        side->generation = generation;
        CONTRACT_TRY(FactorSide(reg, *slot, *side, fill, stream, false, &ran,
                                &used_refactorization));
        if (ran) ++reg.stats.factorizations;
        words[0] = context;
        words[1] = slot->id;
        words[2] = generation;
        words[3] = SideStatus(*side, &status_source);
    } else {
        BatchSide* side = nullptr;
        CONTRACT_TRY(EnsureBatchSide(reg, *slot, fill, stream, &side));
        const int32_t generation = NextGeneration(*slot);
        side->generations.assign(static_cast<size_t>(lanes), generation);
        slot->lane_generations.assign(static_cast<size_t>(lanes), generation);
        int64_t phases = 0;
        CONTRACT_TRY(FactorBatchSide(reg, *slot, *side, &raw, stream, false, nullptr, &phases));
        for (int64_t l = 0; l < lanes; ++l) {
            words[l] = context;
            words[lanes + l] = static_cast<int32_t>(slot->id + l);
            words[2 * lanes + l] = generation;
            words[3 * lanes + l] = side->status[l];
        }
    }
    CONTRACT_TRY(MarkSlotUse(*slot, stream));
    RecordSiteResult(*site, run.run_id, lanes, words);
    int32_t* targets[4] = {out_context->typed_data(), out_slot->typed_data(),
                           out_generation->typed_data(), out_status->typed_data()};
    return WriteResults(stream, *block, first, lanes, targets, 4,
                        status_source != nullptr ? &status_source : nullptr);
}

// Solve: scalar handle leaves solve on a single slot; vector leaves on every lane of one batch
// slot, or lane by lane on single slots. An empty lane (a negative slot id) takes nan and the
// status EMPTY without a cuDSS call. The handle comes from the words the producing call of this
// execution wrote when the program names that call, from the static path when the call passes
// one, and from the operands read on the host otherwise. The values are assembled on the device
// only for a lane that must be factored before it is solved.
template <ffi::DataType T>
ffi::Error SolveImpl(cudaStream_t stream, ffi::RunId run, ffi::Token, ffi::AnyBuffer values,
                     ffi::Buffer<ffi::S32> held_context, ffi::Buffer<ffi::S32> held_slot,
                     ffi::Buffer<ffi::S32> held_generation, ffi::Buffer<T> b,
                     ffi::Buffer<ffi::S32> /*order*/,
                     ffi::Result<ffi::Token>, ffi::ResultBuffer<T> x,
                     ffi::ResultBuffer<ffi::S32> out_status,
                     ffi::Dictionary attrs) {
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    CallAttrs at;
    CONTRACT_TRY(DecodeAttrs(attrs, CallKind::kSolve, &at));
    RawValues raw;
    int64_t value_lanes = 0;
    bool values_batched = false;
    CONTRACT_TRY(RawOperand(values, "a solve", &raw, &value_lanes, &values_batched));
    int64_t lanes = 0;
    CONTRACT_TRY(HandleLanes(held_context.dimensions(), held_slot.dimensions(),
                             held_generation.dimensions(), values_batched ? 2 : 1, &lanes));
    const bool batched = held_slot.dimensions().size() == 1;
    if (value_lanes != lanes) {
        return InvalidCall("the values of a solve carry one lane per handle lane; got " +
                           std::to_string(value_lanes) + " lanes of values for " +
                           std::to_string(lanes) + " handle lanes.");
    }
    if (static_cast<int64_t>(out_status->element_count()) != lanes) {
        return InvalidCall("a solve of " + std::to_string(lanes) +
                           " lanes returns a status of as many entries.");
    }
    if (lanes == 0) return ffi::Error::Success();
    const int64_t total = static_cast<int64_t>(b.element_count());
    if (total % lanes != 0 || static_cast<int64_t>(x->element_count()) != total) {
        return InvalidCall("the right-hand sides of a solve of " + std::to_string(lanes) +
                           " lanes split into lanes of equal size, with a solution of their "
                           "shape; got " +
                           std::to_string(total) + " right-hand-side and " +
                           std::to_string(x->element_count()) + " solution entries.");
    }
    const int64_t lane_total = total / lanes;
    int device = 0;
    CONTRACT_CUDA(cudaGetDevice(&device), "cudaGetDevice");
    const size_t handle_words = 3 * static_cast<size_t>(lanes);
    Staging* block = nullptr;
    CONTRACT_TRY(AcquireStaging(reg, device, handle_words + static_cast<size_t>(lanes), &block));

    Slot* slot = nullptr;
    int32_t generation = 0;
    std::vector<Slot*> singles;
    std::vector<uint8_t> empty;
    bool all_empty = false;
    const int32_t* contexts = nullptr;
    const int32_t* ids = nullptr;
    const int32_t* generations = nullptr;  // per lane, for a batch slot or single slots
    std::vector<int32_t> static_generations;
    bool resolved = false;
    // A solve of a factor that names shared storage uses the storage as it stands when a refactor
    // of that storage ran earlier in this execution of the program (the refactor made it hold the
    // matrix of the factor it returned): its slot is found from the storage key and the context,
    // without reading the handle (see SolveStored). Otherwise (a program that only solves, a
    // deferred linear program, an unknown context) the solve applies the generation rule below.
    bool stored = false;
    if (at.storage != 0) {
        const int32_t context = StorageContext(reg, at);
        if (context >= 0 && reg.released_contexts.count(context) == 0) {
            Slot* found = FindStorageSlot(reg, context, device, at.analysis_id,
                                          get_cuda_data_type<T>(), batched ? lanes : 0,
                                          at.storage);
            if (found != nullptr && found->trusted_run == run.run_id &&
                (batched ? found->batch_primary != nullptr : found->primary != nullptr)) {
                slot = found;
                stored = true;
                resolved = true;
                ++reg.stats.static_dispatches;
            }
        }
    }
    const int32_t* run_words = stored ? nullptr : RunLocalWords(reg, at, run.run_id, lanes);
    if (run_words != nullptr) {
        contexts = run_words;
        ids = run_words + lanes;
        generations = run_words + 2 * lanes;
        ++reg.stats.static_dispatches;
    } else if (!stored && at.StaticPath()) {
        const int32_t context = static_cast<int32_t>(at.static_context);
        CONTRACT_TRY(CheckContext(reg, context));
        Slot* found = ResolveStaticPath(reg, context, at.static_lane, at.static_path);
        if (found != nullptr && found->batch == batched &&
            (!batched || (found->lanes == lanes &&
                          found->lane_generations.size() == static_cast<size_t>(lanes)))) {
            slot = found;
            generation = found->latest_generation;
            static_generations = found->lane_generations;
            generations = static_generations.data();
            resolved = true;
            ++reg.stats.static_dispatches;
        }
    }
    if (!resolved && contexts == nullptr) {
        const size_t bytes = static_cast<size_t>(lanes) * sizeof(int32_t);
        const ReadSpan spans[3] = {{held_context.typed_data(), bytes, 0},
                                   {held_slot.typed_data(), bytes, static_cast<size_t>(lanes)},
                                   {held_generation.typed_data(), bytes,
                                    2 * static_cast<size_t>(lanes)}};
        CONTRACT_TRY(ReadSpans(stream, *block, spans, 3));
        ++reg.stats.host_reads;
        contexts = block->host;
        ids = block->host + lanes;
        generations = block->host + 2 * lanes;
    }
    if (!resolved) {
        if (!batched) {
            if (ids[0] < 0) {
                all_empty = true;
            } else {
                int32_t lane = 0;
                CONTRACT_TRY(FindHeld(reg, contexts[0], ids[0], &slot, &lane));
                if (slot->batch) return UnbatchedLaneError(*slot, lane);
                generation = generations[0];
            }
        } else {
            int64_t first = -1;
            CONTRACT_TRY(ClassifyLanes(reg, lanes, contexts, ids, &slot, &singles,
                                       &empty, &first));
            all_empty = first < 0;
        }
    }

    const auto check = [&](const Slot& s) -> ffi::Error {
        CONTRACT_TRY(CheckAnalysis(s, at.analysis_id));
        CONTRACT_TRY(CheckDtype(s, get_cuda_data_type<T>()));
        CONTRACT_TRY(CheckDevice(s, device));
        CONTRACT_TRY(CheckRaw(*s.analysis, raw));
        if (lane_total % s.analysis->n != 0) {
            return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                              "cuDSS right-hand side has " + std::to_string(lane_total) +
                                  " entries per lane, which is not a whole number of vectors of "
                                  "length " +
                                  std::to_string(s.analysis->n) +
                                  " matching the solution's shape.");
        }
        return ffi::Error::Success();
    };
    int32_t* statuses = block->host + handle_words;
    const size_t vs = sizeof(typename get_native_data_type<T>::type);
    const auto empty_lanes = [&](int64_t first, int64_t count) -> ffi::Error {
        // Every byte 0xFF is a nan in float32 and float64.
        const size_t lane_bytes = static_cast<size_t>(lane_total) * vs;
        if (lane_bytes > 0) {
            CONTRACT_CUDA(cudaMemsetAsync(reinterpret_cast<char*>(x->typed_data()) +
                                              static_cast<size_t>(first) * lane_bytes,
                                          0xFF, static_cast<size_t>(count) * lane_bytes, stream),
                          "cudaMemsetAsync (nan solution of an empty factor)");
        }
        std::fill(statuses + first, statuses + first + count, kEmpty);
        return ffi::Error::Success();
    };
    // A single factor whose verdict is pending reports the device's status: a one-lane call has
    // it written straight into its status output, a lane of several takes it from the factor's
    // device words after the host's words are written.
    std::vector<const int32_t*> sources(static_cast<size_t>(lanes), nullptr);
    int32_t* direct = lanes == 1 ? out_status->typed_data() : nullptr;
    if (all_empty) {
        CONTRACT_TRY(empty_lanes(0, lanes));
    } else if (stored) {
        CONTRACT_TRY(check(*slot));
        CONTRACT_TRY(SolveStored(reg, *slot, raw, b.typed_data(), x->typed_data(), lane_total,
                                 stream, statuses, direct, &sources[0]));
    } else if (!singles.empty()) {
        for (int64_t l = 0; l < lanes; ++l) {
            if (singles[l] == nullptr) {
                CONTRACT_TRY(empty_lanes(l, 1));
                continue;
            }
            Slot& lane_slot = *singles[l];
            CONTRACT_TRY(check(lane_slot));
            const size_t offset = static_cast<size_t>(l * lane_total) * vs;
            CONTRACT_TRY(SolveSingle(
                reg, lane_slot, generations[l], at.transpose, ValueFill::Raw(raw, l),
                reinterpret_cast<const char*>(b.typed_data()) + offset,
                reinterpret_cast<char*>(x->typed_data()) + offset, lane_total, stream,
                &statuses[l], direct, &sources[static_cast<size_t>(l)], false));
        }
    } else {
        CONTRACT_TRY(check(*slot));
        if (!slot->batch) {
            CONTRACT_TRY(SolveSingle(reg, *slot, generation, at.transpose, ValueFill::Raw(raw),
                                     b.typed_data(), x->typed_data(), lane_total, stream,
                                     &statuses[0], direct, &sources[0], false));
        } else {
            CONTRACT_TRY(SolveBatchSlot(reg, *slot, generations, at.transpose,
                                        empty.empty() ? nullptr : &empty, raw, b.typed_data(),
                                        x->typed_data(), lane_total / slot->analysis->n, stream,
                                        statuses, false));
        }
    }
    if (direct != nullptr && sources[0] == direct) return ffi::Error::Success();
    int32_t* targets[1] = {out_status->typed_data()};
    const bool any_source =
        std::any_of(sources.begin(), sources.end(), [](const int32_t* s) { return s != nullptr; });
    return WriteResults(stream, *block, handle_words, lanes, targets, 1,
                        any_source ? sources.data() : nullptr);
}

// The refactor of a factor that names shared storage (the call's storage attribute): the
// storage slot of the key in the factor's context is found without reading the handle (the
// context is a program constant or the one context that declared the key; otherwise the
// predicate and the handle are read first). A filled storage slot is refactored by
// RefactorStored (one read: the merged predicate and value-check words). An unfilled one reads
// the predicate and the handle, and a lane that asks fills it as an empty handle fills a site's
// slot; with no lane asking the handle comes back unchanged.
template <ffi::DataType T>
ffi::Error RefactorStorageCall(cudaStream_t stream, ffi::RunId run, const CallAttrs& at,
                               const RawValues& raw, ffi::AnyBuffer& values,
                               ffi::AnyBuffer& held_values, ffi::Buffer<ffi::PRED>& predicate,
                               ffi::Buffer<ffi::S32>& held_context,
                               ffi::Buffer<ffi::S32>& held_slot,
                               ffi::Buffer<ffi::S32>& held_generation, const KeptValues& kept,
                               int64_t lanes, bool batched, int device, Staging& block,
                               size_t handle_first, size_t result_first,
                               ffi::ResultBuffer<ffi::S32>& out_context,
                               ffi::ResultBuffer<ffi::S32>& out_slot,
                               ffi::ResultBuffer<ffi::S32>& out_generation,
                               ffi::ResultBuffer<ffi::S32>& out_status) {
    Registry& reg = Reg();
    const size_t L = static_cast<size_t>(lanes);
    Site* site = nullptr;
    CONTRACT_TRY(SiteFor(reg, at.site, at.analysis_id, &site));
    CONTRACT_TRY(CheckRaw(*site->analysis, raw));
    const SlotSpec spec = SpecFor<T>(at.transposed, at.symmetric, at.spd);
    RawValues held_raw = raw;
    held_raw.data = held_values.untyped_data();
    int32_t* words = block.host + result_first;
    bool read = false;
    const auto read_handle = [&]() -> ffi::Error {
        const size_t handle_bytes = L * sizeof(int32_t);
        const ReadSpan spans[4] = {{predicate.typed_data(), L, 0},
                                   {held_context.typed_data(), handle_bytes, handle_first},
                                   {held_slot.typed_data(), handle_bytes, handle_first + L},
                                   {held_generation.typed_data(), handle_bytes,
                                    handle_first + 2 * L}};
        CONTRACT_TRY(ReadSpans(stream, block, spans, 4));
        ++reg.stats.host_reads;
        ++reg.stats.predicate_reads;
        read = true;
        return ffi::Error::Success();
    };
    int32_t context = StorageContext(reg, at);
    if (context < 0) {
        CONTRACT_TRY(read_handle());
        context = block.host[handle_first];
    } else {
        ++reg.stats.static_dispatches;
    }
    CONTRACT_TRY(CheckContext(reg, context));
    Slot* slot = FindStorageSlot(reg, context, device, at.analysis_id, spec.dtype,
                                 batched ? lanes : 0, at.storage);
    const bool filled = slot != nullptr && (batched ? slot->batch_primary != nullptr
                                                    : slot->primary != nullptr);
    std::vector<uint8_t> asks(L, 0);
    // The device words that hold a lane's status while its factor's verdict is pending.
    std::vector<const int32_t*> sources(L, nullptr);
    int32_t* targets[4] = {out_context->typed_data(), out_slot->typed_data(),
                           out_generation->typed_data(), out_status->typed_data()};
    // A filled storage writes the kept values in the kernel of its value check, and a one-lane
    // call whose storage keeps its factor has its result written by that kernel too.
    bool kept_written = false;
    bool result_written = false;
    if (filled) {
        CONTRACT_TRY(CheckAnalysis(*slot, at.analysis_id));
        CONTRACT_TRY(CheckDtype(*slot, get_cuda_data_type<T>()));
        CONTRACT_TRY(CheckDevice(*slot, device));
        CONTRACT_TRY(CheckStorageSpec(*slot, spec));
        CONTRACT_TRY(RefactorStored(reg, *slot, context, raw, held_raw,
                                    reinterpret_cast<const uint8_t*>(predicate.typed_data()),
                                    held_slot.typed_data(), lanes, stream, asks.data(), words,
                                    &sources[0], kept.lane_bytes > 0 ? kept.kept : nullptr,
                                    lanes == 1 ? targets : nullptr, &result_written));
        kept_written = true;
    } else {
        if (!read) CONTRACT_TRY(read_handle());
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(block.host);
        const int32_t* contexts = block.host + handle_first;
        const int32_t* ids = block.host + handle_first + L;
        const int32_t* generations = block.host + handle_first + 2 * L;
        bool any = false;
        for (size_t l = 0; l < L; ++l) {
            asks[l] = bytes[l] != 0 ? 1 : 0;
            any = any || asks[l] != 0;
            if (ids[l] >= 0) {
                return ffi::Error(ffi::ErrorCode::kFailedPrecondition,
                                  "cuDSS: this factor names shared storage that its SolveContext "
                                  "does not hold on this device for this analysis, precision and "
                                  "lane count (it was released, or filled on another device); "
                                  "start the carry on this device from empty_factor(..., "
                                  "storage=...) in an open SolveContext.");
            }
        }
        if (!any) {
            for (size_t l = 0; l < L; ++l) {
                words[l] = contexts[l];
                words[L + l] = ids[l];
                words[2 * L + l] = generations[l];
                words[3 * L + l] = kEmpty;
            }
        } else {
            CONTRACT_TRY(StorageSlot(reg, *site, context, at.storage, spec, device, batched, lanes,
                                     &slot));
            CONTRACT_TRY(CheckDevice(*slot, device));
            if (!batched) {
                CONTRACT_TRY(RefactorSingle(reg, *site, nullptr, ids[0], ids[0], context,
                                            generations[0], true, true, ValueFill::Raw(raw), spec,
                                            device, stream, words, &sources[0], slot));
            } else {
                const std::vector<uint8_t> empty(L, 1);
                CONTRACT_TRY(RefactorBatchSlot(reg, *site, nullptr, &empty, context, contexts, ids,
                                               generations, asks.data(), raw, lanes, true, spec,
                                               device, stream, words, slot));
            }
        }
    }
    if (slot != nullptr) {
        site->forward_of_key[SiteKey{context, slot->id}] = slot->id;
        slot->trusted_run = run.run_id;
    }
    if (!kept_written) CONTRACT_TRY(CopyKeptValues(kept, asks.data(), lanes, stream));
    RecordSiteResult(*site, run.run_id, lanes, words);
    if (result_written) return ffi::Error::Success();
    const bool any_source =
        std::any_of(sources.begin(), sources.end(), [](const int32_t* s) { return s != nullptr; });
    return WriteResults(stream, block, result_first, lanes, targets, 4,
                        any_source ? sources.data() : nullptr);
}

// Refactor: the values and the held values (both in the caller's coordinate order, of one dtype
// and shape), the predicate and the held handle; the first result after the token is the kept
// values, aliased to the values operand. The predicate (one entry per lane) is read on the host;
// the held handle comes from the words the producing call of this execution wrote when the
// program names that call, from the static path when the call passes one, and from the operands
// read with the predicate otherwise. A lane that asks is factored from its values, assembled on
// the device straight into the slot's value copy, and keeps them in the result; a lane that does
// not ask keeps its held factor (refactored from the values it holds, under a native vmap whose
// other lanes ask) and gets the held values copied into the result.
template <ffi::DataType T>
ffi::Error RefactorImpl(cudaStream_t stream, ffi::RunId run, ffi::Token, ffi::AnyBuffer values,
                        ffi::AnyBuffer held_values, ffi::Buffer<ffi::PRED> predicate,
                        ffi::Buffer<ffi::S32> held_context, ffi::Buffer<ffi::S32> held_slot,
                        ffi::Buffer<ffi::S32> held_generation, ffi::Buffer<ffi::S32> /*order*/,
                        ffi::Result<ffi::Token>,
                        ffi::Result<ffi::AnyBuffer> kept_values,
                        ffi::ResultBuffer<ffi::S32> out_context,
                        ffi::ResultBuffer<ffi::S32> out_slot,
                        ffi::ResultBuffer<ffi::S32> out_generation,
                        ffi::ResultBuffer<ffi::S32> out_status,
                        ffi::Dictionary attrs) {
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    CallAttrs at;
    CONTRACT_TRY(DecodeAttrs(attrs, CallKind::kRefactor, &at));
    RawValues raw;
    int64_t value_lanes = 0;
    bool values_batched = false;
    CONTRACT_TRY(RawOperand(values, "a refactor", &raw, &value_lanes, &values_batched));
    const size_t bytes = values.size_bytes();
    if (held_values.element_type() != values.element_type() ||
        kept_values->element_type() != values.element_type() ||
        held_values.size_bytes() != bytes || kept_values->size_bytes() != bytes) {
        return InvalidCall("the values, the held values and the kept values of a refactor share "
                           "one dtype and one shape; got " +
                           std::to_string(bytes) + ", " +
                           std::to_string(held_values.size_bytes()) + " and " +
                           std::to_string(kept_values->size_bytes()) + " bytes.");
    }
    int64_t lanes = 0;
    CONTRACT_TRY(HandleLanes(held_context.dimensions(), held_slot.dimensions(),
                             held_generation.dimensions(), values_batched ? 2 : 1, &lanes));
    const bool batched = held_slot.dimensions().size() == 1;
    const int64_t results[4] = {static_cast<int64_t>(out_context->element_count()),
                                static_cast<int64_t>(out_slot->element_count()),
                                static_cast<int64_t>(out_generation->element_count()),
                                static_cast<int64_t>(out_status->element_count())};
    bool shapes_agree =
        value_lanes == lanes && static_cast<int64_t>(predicate.element_count()) == lanes;
    for (int64_t count : results) shapes_agree = shapes_agree && count == lanes;
    if (!shapes_agree) {
        return InvalidCall("a refactor of " + std::to_string(lanes) +
                           " lanes takes values and a predicate of as many lanes and returns "
                           "handle and status leaves of as many entries; got " +
                           std::to_string(value_lanes) + " lanes of values and a predicate of " +
                           std::to_string(predicate.element_count()) + ".");
    }
    if (lanes == 0) return ffi::Error::Success();
    KeptValues kept;
    kept.raw = static_cast<const char*>(values.untyped_data());
    kept.held = static_cast<const char*>(held_values.untyped_data());
    kept.kept = static_cast<char*>(kept_values->untyped_data());
    kept.lane_bytes = bytes / static_cast<size_t>(lanes);
    int device = 0;
    CONTRACT_CUDA(cudaGetDevice(&device), "cudaGetDevice");
    const size_t L = static_cast<size_t>(lanes);
    const size_t predicate_words = (L + sizeof(int32_t) - 1) / sizeof(int32_t);
    const size_t handle_first = predicate_words;
    const size_t result_first = predicate_words + 3 * L;
    Staging* block = nullptr;
    CONTRACT_TRY(AcquireStaging(reg, device, result_first + 4 * L, &block));

    if (at.storage != 0) {
        return RefactorStorageCall<T>(stream, run, at, raw, values, held_values, predicate,
                                      held_context, held_slot, held_generation, kept, lanes,
                                      batched, device, *block, handle_first, result_first,
                                      out_context, out_slot, out_generation, out_status);
    }
    Slot* held = nullptr;
    int32_t held_generation_value = 0;
    int32_t context = -1;
    std::vector<Slot*> singles;
    std::vector<uint8_t> empty;
    const int32_t* contexts = nullptr;
    const int32_t* ids = nullptr;
    const int32_t* generations = nullptr;
    std::vector<int32_t> static_generations;
    bool resolved = false;
    if (const int32_t* words = RunLocalWords(reg, at, run.run_id, lanes)) {
        contexts = words;
        ids = words + lanes;
        generations = words + 2 * lanes;
        ++reg.stats.static_dispatches;
    } else if (at.StaticPath()) {
        context = static_cast<int32_t>(at.static_context);
        CONTRACT_TRY(CheckContext(reg, context));
        Slot* found = ResolveStaticPath(reg, context, at.static_lane, at.static_path);
        if (found != nullptr && found->batch == batched &&
            (!batched || (found->lanes == lanes &&
                          found->lane_generations.size() == static_cast<size_t>(lanes)))) {
            held = found;
            held_generation_value = found->latest_generation;
            static_generations = found->lane_generations;
            generations = static_generations.data();
            resolved = true;
            ++reg.stats.static_dispatches;
        }
    }
    if (resolved || contexts != nullptr) {
        const ReadSpan spans[1] = {{predicate.typed_data(), L, 0}};
        // A held single factor whose verdict is pending is settled in the same wait as the
        // predicate: the call settles it whatever the predicate says (RefactorSingle).
        Side* pending = resolved && !batched && held != nullptr && held->primary &&
                                held->primary->pending && held->device == device
                            ? held->primary.get()
                            : nullptr;
        if (pending != nullptr) {
            CONTRACT_TRY(BindSlot(*held, stream));
            CONTRACT_TRY(ReadSpans(stream, *block, spans, 1, false));
            CONTRACT_TRY(SettleSide(reg, *held, *pending, stream));
            CONTRACT_TRY(MarkSlotUse(*held, stream));
        } else {
            CONTRACT_TRY(ReadSpans(stream, *block, spans, 1));
        }
        ++reg.stats.predicate_reads;
    } else {
        const size_t handle_bytes = L * sizeof(int32_t);
        const ReadSpan spans[4] = {{predicate.typed_data(), L, 0},
                                   {held_context.typed_data(), handle_bytes, handle_first},
                                   {held_slot.typed_data(), handle_bytes, handle_first + L},
                                   {held_generation.typed_data(), handle_bytes,
                                    handle_first + 2 * L}};
        CONTRACT_TRY(ReadSpans(stream, *block, spans, 4));
        ++reg.stats.host_reads;
        ++reg.stats.predicate_reads;
        contexts = block->host + handle_first;
        ids = block->host + handle_first + L;
        generations = block->host + handle_first + 2 * L;
    }
    if (!resolved) {
        int64_t first = 0;
        if (!batched) {
            if (ids[0] >= 0) {
                int32_t lane = 0;
                CONTRACT_TRY(FindHeld(reg, contexts[0], ids[0], &held, &lane));
                if (held->batch) return UnbatchedLaneError(*held, lane);
            }
        } else {
            CONTRACT_TRY(ClassifyLanes(reg, lanes, contexts, ids, &held, &singles,
                                       &empty, &first));
            if (first < 0) first = 0;
        }
        context = contexts[first];
        held_generation_value = generations[first];
    }
    const uint8_t* asks = reinterpret_cast<const uint8_t*>(block->host);
    Site* site = nullptr;
    CONTRACT_TRY(SiteFor(reg, at.site, at.analysis_id, &site));
    CONTRACT_TRY(CheckRaw(*site->analysis, raw));
    const SlotSpec spec = SpecFor<T>(at.transposed, at.symmetric, at.spd);
    const auto check = [&](const Slot& s) -> ffi::Error {
        CONTRACT_TRY(CheckAnalysis(s, at.analysis_id));
        CONTRACT_TRY(CheckDtype(s, get_cuda_data_type<T>()));
        CONTRACT_TRY(CheckDevice(s, device));
        return ffi::Error::Success();
    };
    int32_t* words = block->host + result_first;
    // The device words that hold a lane's status while its factor's verdict is pending.
    std::vector<const int32_t*> sources(static_cast<size_t>(lanes), nullptr);
    if (!singles.empty()) {
        for (int64_t l = 0; l < lanes; ++l) {
            Slot* lane_held = singles[l];
            if (lane_held != nullptr) CONTRACT_TRY(check(*lane_held));
            int32_t out[4] = {0, 0, 0, 0};
            // An empty lane's slot at the site is keyed by its slot id when that names the lane
            // (a negative id other than -1), by its lane otherwise.
            const int64_t empty_key = ids[l] < kEmptySlot ? ids[l] : kEmptySlot - 1 - l;
            CONTRACT_TRY(RefactorSingle(reg, *site, lane_held, ids[l], empty_key, contexts[l],
                                        generations[l], asks[l] != 0, at.adopt,
                                        ValueFill::Raw(raw, l), spec, device, stream, out,
                                        &sources[static_cast<size_t>(l)]));
            for (int k = 0; k < 4; ++k) words[k * lanes + l] = out[k];
        }
    } else if (!batched) {
        if (held != nullptr) CONTRACT_TRY(check(*held));
        const int32_t empty_id = ids != nullptr ? ids[0] : kEmptySlot;
        CONTRACT_TRY(RefactorSingle(reg, *site, held, empty_id, empty_id, context,
                                    held_generation_value, asks[0] != 0, at.adopt,
                                    ValueFill::Raw(raw), spec, device, stream, words,
                                    &sources[0]));
    } else {
        if (held != nullptr) CONTRACT_TRY(check(*held));
        CONTRACT_TRY(RefactorBatchSlot(reg, *site, held, empty.empty() ? nullptr : &empty,
                                       context, contexts, ids, generations, asks, raw, lanes,
                                       at.adopt, spec, device, stream, words));
    }
    CONTRACT_TRY(CopyKeptValues(kept, asks, lanes, stream));
    RecordSiteResult(*site, run.run_id, lanes, words);
    int32_t* targets[4] = {out_context->typed_data(), out_slot->typed_data(),
                           out_generation->typed_data(), out_status->typed_data()};
    const bool any_source =
        std::any_of(sources.begin(), sources.end(), [](const int32_t* s) { return s != nullptr; });
    return WriteResults(stream, *block, result_first, lanes, targets, 4,
                        any_source ? sources.data() : nullptr);
}

// Invalidate: the slot each handle names stands for no handle any more. A batched handle names
// every lane of one batch slot (invalidated at once) or single slots only; a scalar handle that
// names one lane of a batch slot fails, as a solve or refactor of it does. An empty lane (a
// negative slot id) names nothing and is left alone. The handle comes from the words the
// producing call of this execution wrote when the program names that call, from the static path
// when the call passes one, and from the operands read on the host otherwise.
template <ffi::DataType T>
ffi::Error InvalidateImpl(cudaStream_t stream, ffi::RunId run, ffi::Token,
                          ffi::Buffer<ffi::S32> held_context, ffi::Buffer<ffi::S32> held_slot,
                          ffi::Buffer<ffi::S32> held_generation, ffi::Buffer<ffi::S32> /*order*/,
                          ffi::Result<ffi::Token>, ffi::ResultBuffer<ffi::S32> /*order_out*/,
                          ffi::Dictionary attrs) {
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    CallAttrs at;
    CONTRACT_TRY(DecodeAttrs(attrs, CallKind::kInvalidate, &at));
    int64_t lanes = 0;
    CONTRACT_TRY(HandleLanes(held_context.dimensions(), held_slot.dimensions(),
                             held_generation.dimensions(), -1, &lanes));
    const bool batched = held_slot.dimensions().size() == 1;
    std::vector<Slot*> named;
    bool resolved = false;
    const int32_t* contexts = nullptr;
    const int32_t* ids = nullptr;
    if (lanes > 0) {
        if (at.storage != 0) {
            // The storage the factor names on this call's device, found without reading the
            // handle when the context is known (StorageContext); an unfilled storage has nothing
            // to invalidate.
            int device = 0;
            CONTRACT_CUDA(cudaGetDevice(&device), "cudaGetDevice");
            int32_t context = StorageContext(reg, at);
            if (context < 0) {
                Staging* block = nullptr;
                CONTRACT_TRY(AcquireStaging(reg, device, 1, &block));
                const ReadSpan spans[1] = {{held_context.typed_data(), sizeof(int32_t), 0}};
                CONTRACT_TRY(ReadSpans(stream, *block, spans, 1));
                ++reg.stats.host_reads;
                context = block->host[0];
            } else {
                ++reg.stats.static_dispatches;
            }
            CONTRACT_TRY(CheckContext(reg, context));
            Slot* found = FindStorageSlot(reg, context, device, at.analysis_id,
                                          get_cuda_data_type<T>(), batched ? lanes : 0,
                                          at.storage);
            if (found != nullptr) named.push_back(found);
            resolved = true;
        } else if (const int32_t* words = RunLocalWords(reg, at, run.run_id, lanes)) {
            contexts = words;
            ids = words + lanes;
            ++reg.stats.static_dispatches;
        } else if (at.StaticPath()) {
            const int32_t context = static_cast<int32_t>(at.static_context);
            CONTRACT_TRY(CheckContext(reg, context));
            Slot* found = ResolveStaticPath(reg, context, at.static_lane, at.static_path);
            if (found != nullptr && found->batch == batched &&
                (!batched || found->lanes == lanes)) {
                named.push_back(found);
                resolved = true;
                ++reg.stats.static_dispatches;
            }
        }
    }
    if (lanes > 0 && !resolved && contexts == nullptr) {
        int device = 0;
        CONTRACT_CUDA(cudaGetDevice(&device), "cudaGetDevice");
        const size_t L = static_cast<size_t>(lanes);
        Staging* block = nullptr;
        CONTRACT_TRY(AcquireStaging(reg, device, 3 * L, &block));
        const size_t bytes = L * sizeof(int32_t);
        const ReadSpan spans[3] = {{held_context.typed_data(), bytes, 0},
                                   {held_slot.typed_data(), bytes, L},
                                   {held_generation.typed_data(), bytes, 2 * L}};
        CONTRACT_TRY(ReadSpans(stream, *block, spans, 3));
        ++reg.stats.host_reads;
        contexts = block->host;
        ids = block->host + L;
    }
    if (lanes > 0 && !resolved) {
        if (!batched) {
            if (ids[0] >= 0) {
                Slot* slot = nullptr;
                int32_t lane = 0;
                CONTRACT_TRY(FindHeld(reg, contexts[0], ids[0], &slot, &lane));
                if (slot->batch) return UnbatchedLaneError(*slot, lane);
                named.push_back(slot);
            }
        } else {
            Slot* batch = nullptr;
            std::vector<Slot*> singles;
            std::vector<uint8_t> empty;
            int64_t first = -1;
            CONTRACT_TRY(ClassifyLanes(reg, lanes, contexts, ids, &batch, &singles,
                                       &empty, &first));
            if (batch != nullptr) named.push_back(batch);
            for (Slot* slot : singles) {
                if (slot != nullptr && std::find(named.begin(), named.end(), slot) == named.end()) {
                    named.push_back(slot);
                }
            }
        }
    }
    for (Slot* slot : named) {
        CONTRACT_TRY(CheckAnalysis(*slot, at.analysis_id));
        CONTRACT_TRY(CheckDtype(*slot, get_cuda_data_type<T>()));
        InvalidateSlot(*slot);
    }
    ++reg.stats.invalidations;
    return ffi::Error::Success();
}

}  // namespace cudss_contract

#define DIFF_SOUP_CUDSS_CONTRACT_HANDLERS(Suffix, DataType)                                   \
    XLA_FFI_DEFINE_HANDLER(kCudssContractFactor##Suffix, cudss_contract::FactorImpl<DataType>, \
                           ffi::Ffi::Bind()                                                   \
                               .Ctx<ffi::PlatformStream<cudaStream_t>>()                      \
                               .Ctx<ffi::RunId>()                                             \
                               .Arg<ffi::Token>()               /* XLA token in */             \
                               .Arg<ffi::AnyBuffer>()           /* coordinate values */        \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* lane token */               \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* context id */               \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* order word in */            \
                               .Ret<ffi::Token>()               /* XLA token out */            \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: context id */       \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: slot */             \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: generation */       \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* status */                   \
                               .Attrs());                                                     \
    XLA_FFI_DEFINE_HANDLER(kCudssContractSolve##Suffix, cudss_contract::SolveImpl<DataType>,   \
                           ffi::Ffi::Bind()                                                   \
                               .Ctx<ffi::PlatformStream<cudaStream_t>>()                      \
                               .Ctx<ffi::RunId>()                                             \
                               .Arg<ffi::Token>()               /* XLA token in */             \
                               .Arg<ffi::AnyBuffer>()           /* coordinate values */        \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: context id */       \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: slot */             \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: generation */       \
                               .Arg<ffi::Buffer<DataType>>()    /* right-hand sides */         \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* order word in */            \
                               .Ret<ffi::Token>()               /* XLA token out */            \
                               .Ret<ffi::Buffer<DataType>>()    /* solution */                 \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* status */                   \
                               .Attrs());                                                     \
    XLA_FFI_DEFINE_HANDLER(kCudssContractRefactor##Suffix,                                    \
                           cudss_contract::RefactorImpl<DataType>,                            \
                           ffi::Ffi::Bind()                                                   \
                               .Ctx<ffi::PlatformStream<cudaStream_t>>()                      \
                               .Ctx<ffi::RunId>()                                             \
                               .Arg<ffi::Token>()               /* XLA token in */             \
                               .Arg<ffi::AnyBuffer>()           /* coordinate values (aliased out) */ \
                               .Arg<ffi::AnyBuffer>()           /* held values */              \
                               .Arg<ffi::Buffer<ffi::PRED>>()   /* predicate */                \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* held handle: context id */  \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* held handle: slot */        \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* held handle: generation */  \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* order word in */            \
                               .Ret<ffi::Token>()               /* XLA token out */            \
                               .Ret<ffi::AnyBuffer>()           /* kept values */              \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: context id */       \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: slot */             \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* handle: generation */       \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* status */                   \
                               .Attrs());                                                     \
    XLA_FFI_DEFINE_HANDLER(kCudssContractInvalidate##Suffix,                                  \
                           cudss_contract::InvalidateImpl<DataType>,                          \
                           ffi::Ffi::Bind()                                                   \
                               .Ctx<ffi::PlatformStream<cudaStream_t>>()                      \
                               .Ctx<ffi::RunId>()                                             \
                               .Arg<ffi::Token>()               /* XLA token in */             \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: context id */       \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: slot */             \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* handle: generation */       \
                               .Arg<ffi::Buffer<ffi::S32>>()    /* order word in */            \
                               .Ret<ffi::Token>()               /* XLA token out */            \
                               .Ret<ffi::Buffer<ffi::S32>>()    /* order word out (aliased) */ \
                               .Attrs());

DIFF_SOUP_CUDSS_CONTRACT_HANDLERS(F64, ffi::F64)
DIFF_SOUP_CUDSS_CONTRACT_HANDLERS(F32, ffi::F32)

// --------------------------------------------------------------------------
// Python module functions
// --------------------------------------------------------------------------

namespace cudss_contract {

using IndexArray = nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ByteArray = nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Check the CSR pattern of an n x n operator: n + 1 non-decreasing offsets from zero to the
// number of columns, columns inside the operator and no column twice in a row.
void CheckPattern(int64_t n, const IndexArray& offsets, const IndexArray& columns) {
    const std::string label = "cuDSS analysis: the pattern";
    if (static_cast<int64_t>(offsets.shape(0)) != n + 1) {
        throw std::invalid_argument(label + " needs n + 1 = " + std::to_string(n + 1) +
                                    " offsets; got " + std::to_string(offsets.shape(0)) + ".");
    }
    const int32_t* off = offsets.data();
    const int32_t* col = columns.data();
    const int64_t nse = static_cast<int64_t>(columns.shape(0));
    if (off[0] != 0 || off[n] != nse) {
        throw std::invalid_argument(label + " offsets must run from 0 to the number of columns (" +
                                    std::to_string(nse) + "); got " + std::to_string(off[0]) +
                                    " to " + std::to_string(off[n]) + ".");
    }
    std::vector<int32_t> row;
    for (int64_t r = 0; r < n; ++r) {
        if (off[r + 1] < off[r]) {
            throw std::invalid_argument(label + " offsets decrease at row " + std::to_string(r) +
                                        ".");
        }
        row.assign(col + off[r], col + off[r + 1]);
        for (int32_t c : row) {
            if (c < 0 || c >= n) {
                throw std::invalid_argument(label + " has column " + std::to_string(c) +
                                            " in row " + std::to_string(r) + ", outside the " +
                                            std::to_string(n) + " x " + std::to_string(n) +
                                            " operator.");
            }
        }
        std::sort(row.begin(), row.end());
        if (std::adjacent_find(row.begin(), row.end()) != row.end()) {
            throw std::invalid_argument(label + " lists a column twice in row " +
                                        std::to_string(r) +
                                        "; pass the assembled pattern with unique entries.");
        }
    }
}

// Check one assembly map (assembly_kernels.h) of a pattern with `nnz` coordinate entries and
// `nse` CSR positions: nse + 1 non-decreasing offsets from zero to nnz, every coordinate entry
// listed once (as e, or as -(e + 1) for an entry in a Dirichlet row), and a diagonal marker per
// position or none at all.
void CheckAssemblyMap(int64_t nnz, int64_t nse, const IndexArray& offsets,
                      const IndexArray& entries, const ByteArray& diagonal_add) {
    const std::string label = "cuDSS analysis: the assembly map";
    if (static_cast<int64_t>(offsets.shape(0)) != nse + 1) {
        throw std::invalid_argument(label + " needs nse + 1 = " + std::to_string(nse + 1) +
                                    " offsets; got " + std::to_string(offsets.shape(0)) + ".");
    }
    const int32_t* off = offsets.data();
    if (off[0] != 0 || off[nse] != nnz) {
        throw std::invalid_argument(label + " offsets must run from 0 to the number of "
                                    "coordinate entries (" +
                                    std::to_string(nnz) + "); got " + std::to_string(off[0]) +
                                    " to " + std::to_string(off[nse]) + ".");
    }
    for (int64_t p = 0; p < nse; ++p) {
        if (off[p + 1] < off[p]) {
            throw std::invalid_argument(label + " offsets decrease at position " +
                                        std::to_string(p) + ".");
        }
    }
    if (static_cast<int64_t>(entries.shape(0)) != nnz) {
        throw std::invalid_argument(label + " needs one entry per coordinate (" +
                                    std::to_string(nnz) + "); got " +
                                    std::to_string(entries.shape(0)) + ".");
    }
    std::vector<uint8_t> seen(static_cast<size_t>(nnz), 0);
    const int32_t* ent = entries.data();
    for (int64_t k = 0; k < nnz; ++k) {
        const int64_t e = ent[k] >= 0 ? ent[k] : -static_cast<int64_t>(ent[k]) - 1;
        if (e < 0 || e >= nnz) {
            throw std::invalid_argument(label + " names coordinate entry " + std::to_string(e) +
                                        ", outside the " + std::to_string(nnz) + " entries.");
        }
        if (seen[static_cast<size_t>(e)] != 0) {
            throw std::invalid_argument(label + " lists coordinate entry " + std::to_string(e) +
                                        " twice; every entry adds into one CSR position.");
        }
        seen[static_cast<size_t>(e)] = 1;
    }
    const int64_t add = static_cast<int64_t>(diagonal_add.shape(0));
    if (add != 0 && add != nse) {
        throw std::invalid_argument(label + " needs a diagonal marker per CSR position (" +
                                    std::to_string(nse) + ") or none; got " +
                                    std::to_string(add) + ".");
    }
}

int64_t AnalysisCreate(int64_t n, IndexArray csr_offsets, IndexArray csr_columns, int64_t nnz,
                       IndexArray gather_offsets, IndexArray gather_entries,
                       ByteArray diagonal_add, int64_t reorder_alg, int64_t hybrid_mode,
                       int64_t hybrid_device_memory_limit_mb, int64_t deterministic,
                       int64_t ir_nsteps, int64_t symmetric_factorization, int64_t zero_fill,
                       int64_t downcast_factors, int64_t emulation_mantissa_bits,
                       int64_t batch_layout) {
    if (n <= 0 || n > std::numeric_limits<int32_t>::max() - 1) {
        throw std::invalid_argument("cuDSS analysis needs 0 < n < 2**31 - 1; got n = " +
                                    std::to_string(n) + ".");
    }
    const int64_t nse = static_cast<int64_t>(csr_columns.shape(0));
    if (nse <= 0) {
        throw std::invalid_argument("cuDSS analysis needs a non-empty pattern; got " +
                                    std::to_string(nse) + " entries.");
    }
    if (nnz < nse || nnz > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("cuDSS analysis needs at least one coordinate entry per "
                                    "pattern position and fewer than 2**31; got " +
                                    std::to_string(nnz) + " entries for " +
                                    std::to_string(nse) + " positions.");
    }
    if (reorder_alg < -1 || reorder_alg > static_cast<int64_t>(CUDSS_REORDERING_ALG_NONE)) {
        throw std::invalid_argument(
            "cuDSS reordering algorithm id must be -1 (the library default) or a "
            "cudssReorderingAlg_t value from 0 to " +
            std::to_string(static_cast<int>(CUDSS_REORDERING_ALG_NONE)) + "; got " +
            std::to_string(reorder_alg) + ".");
    }
    if (hybrid_mode != 0 && hybrid_mode != 1) {
        throw std::invalid_argument("cuDSS hybrid_mode must be 0 or 1; got " +
                                    std::to_string(hybrid_mode) + ".");
    }
    if (deterministic != 0 && deterministic != 1) {
        throw std::invalid_argument("cuDSS deterministic must be 0 or 1; got " +
                                    std::to_string(deterministic) + ".");
    }
    if (ir_nsteps < -1 || ir_nsteps > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "cuDSS ir_nsteps must be a non-negative int, or -1 for the factor's default; got " +
                                    std::to_string(ir_nsteps) + ".");
    }
    if (zero_fill < -1 || zero_fill > 1) {
        throw std::invalid_argument(
            "cuDSS zero_fill must be 0 or 1, or -1 for the factor's default; got " +
                                    std::to_string(zero_fill) + ".");
    }
    if (downcast_factors < kDowncastDefault ||
        downcast_factors > static_cast<int64_t>(CUDSS_DOWNCAST_FACTORS_MODE_KEEP_STRICT_DIAGONAL)) {
        throw std::invalid_argument(
            "cuDSS downcast_factors must be -1 (the factor's default) or a "
            "cudssDowncastFactorsMode_t value from 0 to " +
            std::to_string(static_cast<int>(CUDSS_DOWNCAST_FACTORS_MODE_KEEP_STRICT_DIAGONAL)) +
            "; got " + std::to_string(downcast_factors) + ".");
    }
    if (emulation_mantissa_bits < 0 || emulation_mantissa_bits > 64) {
        throw std::invalid_argument(
            "cuDSS emulation_mantissa_bits must be 0 (emulation off) or a bit count up to 64; "
            "got " + std::to_string(emulation_mantissa_bits) + ".");
    }
    if (batch_layout != kBatchStacked && batch_layout != kBatchUniform) {
        throw std::invalid_argument("cuDSS batch_layout must be " + std::to_string(kBatchStacked) +
                                    " (stacked) or " + std::to_string(kBatchUniform) +
                                    " (uniform); got " + std::to_string(batch_layout) + ".");
    }
    auto a = std::make_shared<Analysis>();
    {
        nb::gil_scoped_release release;
        CheckPattern(n, csr_offsets, csr_columns);
        CheckAssemblyMap(nnz, nse, gather_offsets, gather_entries, diagonal_add);
        a->n = static_cast<int32_t>(n);
        a->nse = nse;
        a->nnz = nnz;
        a->offsets.assign(csr_offsets.data(), csr_offsets.data() + n + 1);
        a->columns.assign(csr_columns.data(), csr_columns.data() + nse);
        a->gather_offsets.assign(gather_offsets.data(), gather_offsets.data() + nse + 1);
        a->gather_entries.assign(gather_entries.data(), gather_entries.data() + nnz);
        a->diagonal_add.assign(diagonal_add.data(), diagonal_add.data() + diagonal_add.shape(0));
        a->reorder_alg = static_cast<int>(reorder_alg);
        a->hybrid_mode = static_cast<int>(hybrid_mode);
        a->hybrid_device_memory_limit_mb = std::max<int64_t>(hybrid_device_memory_limit_mb, 0);
        a->deterministic = static_cast<int>(deterministic);
        a->ir_nsteps = static_cast<int>(ir_nsteps);
        a->symmetric_factorization = symmetric_factorization != 0;
        a->zero_fill = static_cast<int>(zero_fill);
        a->downcast_factors = static_cast<int>(downcast_factors);
        a->emulation_mantissa_bits = static_cast<int>(emulation_mantissa_bits);
        a->batch_layout = static_cast<int>(batch_layout);
        std::lock_guard<std::mutex> lock(ContractMutex());
        Registry& reg = Reg();
        a->id = reg.next_analysis++;
        reg.analyses[a->id] = a;
    }
    return a->id;
}

void AnalysisRelease(int64_t analysis_id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    Reg().analyses.erase(analysis_id);  // sites and slots that use it keep it alive
}

std::shared_ptr<Analysis> FindAnalysis(Registry& reg, int64_t analysis_id) {
    auto it = reg.analyses.find(analysis_id);
    if (it == reg.analyses.end()) {
        throw std::invalid_argument("cuDSS analysis " + std::to_string(analysis_id) +
                                    " does not exist in this process; analyse the pattern "
                                    "again.");
    }
    return it->second;
}

// Make every factor of the analysis spread over a model group with cuDSS MG: `mg_devices` lists
// the CUDA ordinals of this process's model groups, `mg_group_size` consecutive ordinals per
// group, and each single slot factors over the group that holds the device of its first call
// (see "Factors over a model group"). Allowed only before the analysis serves any factor; the
// device set is checked when a slot is created.
void AnalysisSetModelGroups(int64_t analysis_id, IndexArray mg_devices, int64_t mg_group_size) {
    if (mg_group_size < 2) {
        throw std::invalid_argument("cuDSS model groups need at least 2 devices per group; got " +
                                    std::to_string(mg_group_size) + ".");
    }
    const int64_t count = static_cast<int64_t>(mg_devices.shape(0));
    if (count == 0 || count % mg_group_size != 0) {
        throw std::invalid_argument("cuDSS model groups: mg_devices holds " +
                                    std::to_string(count) +
                                    " CUDA ordinals, not a whole number of groups of " +
                                    std::to_string(mg_group_size) + " devices.");
    }
    std::vector<int> devices(mg_devices.data(), mg_devices.data() + count);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    std::shared_ptr<Analysis> a = FindAnalysis(Reg(), analysis_id);
    if (!a->contexts.empty() || a->DeviceResident()) {
        throw std::invalid_argument("cuDSS analysis " + std::to_string(analysis_id) +
                                    " already serves factors; set its model groups before its "
                                    "first factor.");
    }
    a->mg_devices = std::move(devices);
    a->mg_group_size = mg_group_size;
}

nb::dict AnalysisInfo(int64_t analysis_id) {
    Analysis copy_view;  // filled under the lock, read without it
    bool resident = false;
    int64_t contexts = 0;
    int64_t pattern_bytes = 0;
    int64_t nd_tree_bytes = 0;
    int tree_seed = -1;  // -1 no unmatched cold analysis yet, 1 tree kept, 0 none reported
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(ContractMutex());
        std::shared_ptr<Analysis> a = FindAnalysis(Reg(), analysis_id);
        resident = a->DeviceResident();
        contexts = static_cast<int64_t>(a->contexts.size());
        pattern_bytes = a->PatternBytes();
        copy_view.cold_analyses = a->cold_analyses;
        copy_view.seeded_analyses = a->seeded_analyses;
        copy_view.seed_refusals = a->seed_refusals;
        copy_view.tree_seeded_analyses = a->tree_seeded_analyses;
        nd_tree_bytes = static_cast<int64_t>(a->nd_tree.size());
        if (static_cast<int32_t>(a->tree_permutation.size()) == a->n) {
            tree_seed = a->nd_tree.empty() ? 0 : 1;
        }
        copy_view.lu_nnz = a->lu_nnz;
        copy_view.pivot_ratio = a->pivot_ratio;
        copy_view.lu_nnz_batch = a->lu_nnz_batch;
        copy_view.analysis_seconds_cold = a->analysis_seconds_cold;
        copy_view.analysis_seconds_seeded_last = a->analysis_seconds_seeded_last;
        copy_view.hybrid_device_memory_min = a->hybrid_device_memory_min;
        copy_view.n = a->n;
        copy_view.nse = a->nse;
        copy_view.batch_analyses = a->batch_analyses;
        copy_view.batch_seeded = a->batch_seeded;
        copy_view.analysis_seconds_batch_last = a->analysis_seconds_batch_last;
        copy_view.batch_layout = a->batch_layout;
    }
    nb::dict d;
    d["device_resident"] = resident;
    d["contexts"] = contexts;
    d["cold_analyses"] = copy_view.cold_analyses;
    d["seeded_analyses"] = copy_view.seeded_analyses;
    d["seed_refusals"] = copy_view.seed_refusals;
    d["tree_seeded_analyses"] = copy_view.tree_seeded_analyses;
    d["nd_tree_bytes"] = nd_tree_bytes;
    if (tree_seed < 0) {
        d["tree_seed"] = nb::none();
    } else {
        d["tree_seed"] = tree_seed == 1;
    }
    if (copy_view.lu_nnz < 0) {
        d["lu_nnz_forward"] = nb::none();
    } else {
        d["lu_nnz_forward"] = copy_view.lu_nnz;
    }
    if (copy_view.pivot_ratio < 0.0) {
        d["pivot_ratio_forward"] = nb::none();
    } else {
        d["pivot_ratio_forward"] = copy_view.pivot_ratio;
    }
    d["analysis_seconds_cold"] = copy_view.analysis_seconds_cold;
    d["analysis_seconds_seeded_last"] = copy_view.analysis_seconds_seeded_last;
    d["pattern_bytes"] = pattern_bytes;
    if (copy_view.hybrid_device_memory_min < 0) {
        d["hybrid_device_memory_min"] = nb::none();
    } else {
        d["hybrid_device_memory_min"] = copy_view.hybrid_device_memory_min;
    }
    d["n"] = static_cast<int64_t>(copy_view.n);
    d["nse"] = copy_view.nse;
    d["batch_analyses"] = copy_view.batch_analyses;
    if (copy_view.batch_seeded < 0) {
        d["batch_seeded"] = nb::none();
    } else {
        d["batch_seeded"] = copy_view.batch_seeded == 1;
    }
    d["analysis_seconds_batch_last"] = copy_view.analysis_seconds_batch_last;
    d["batch_layout"] = copy_view.batch_layout == kBatchUniform ? "uniform" : "stacked";
    if (copy_view.lu_nnz_batch < 0) {
        d["lu_nnz_batch"] = nb::none();
    } else {
        d["lu_nnz_batch"] = copy_view.lu_nnz_batch;
    }
    return d;
}

int64_t SiteCreate(int64_t analysis_id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    std::shared_ptr<Analysis> a = FindAnalysis(reg, analysis_id);
    while (reg.sites.count(reg.next_site) != 0) ++reg.next_site;
    auto site = std::make_unique<Site>();
    site->id = reg.next_site++;
    site->analysis = std::move(a);
    const int64_t id = site->id;
    reg.sites.emplace(id, std::move(site));
    return id;
}

void StorageDeclare(int64_t analysis_id, int64_t context_id, int64_t storage) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    if (storage == 0) return;
    reg.storage_contexts[{analysis_id, storage}].insert(static_cast<int32_t>(context_id));
}

void ContextRelease(int64_t context_id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    Registry& reg = Reg();
    const int32_t context = static_cast<int32_t>(context_id);
    int current = -1;
    cudaGetDevice(&current);
    // Wait for the work queued on every device that holds a slot of the context before its
    // cuDSS objects and buffers are freed.
    std::set<int> devices;
    for (const auto& entry : reg.slots) {
        if (entry.second->context == context) devices.insert(entry.second->device);
    }
    for (int device : devices) {
        if (cudaSetDevice(device) == cudaSuccess) cudaDeviceSynchronize();
    }
    std::set<Analysis*> touched;
    for (auto it = reg.slots.begin(); it != reg.slots.end();) {
        Slot& slot = *it->second;
        if (slot.context != context) {
            ++it;
            continue;
        }
        if (Site* site = FindSite(reg, slot.site); site != nullptr && slot.storage == 0) {
            site->slot_of_key.erase(slot.key);
        }
        slot.analysis->contexts.erase(context);
        touched.insert(slot.analysis.get());
        it = reg.slots.erase(it);  // the slot frees its cuDSS objects and buffers
    }
    for (auto it = reg.storage_slots.begin(); it != reg.storage_slots.end();) {
        it = std::get<0>(it->first) == context ? reg.storage_slots.erase(it) : std::next(it);
    }
    for (auto& entry : reg.storage_contexts) entry.second.erase(context);
    for (auto& entry : reg.sites) {
        auto& forwards = entry.second->forward_of_key;
        for (auto it = forwards.begin(); it != forwards.end();) {
            it = it->first.first == context ? forwards.erase(it) : std::next(it);
        }
    }
    for (Analysis* a : touched) {
        if (a->contexts.empty() && a->FreeDevice()) {
            --reg.device_analyses;
        }
    }
    reg.released_contexts.insert(context);
    if (current >= 0) cudaSetDevice(current);
    (void)cudaGetLastError();
}

int64_t ResidentBytes(int64_t context_id) {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(ContractMutex());
    int64_t total = 0;
    for (const auto& entry : Reg().slots) {
        const Slot& slot = *entry.second;
        if (slot.context != context_id) continue;
        if (slot.primary) total += slot.primary->ResidentBytes();
        if (slot.batch_primary) total += slot.batch_primary->ResidentBytes();
    }
    return total;
}

// Settle every pending verdict of a single side (one wait each, on the stream of the slot's
// latest call), so the counters include them. Throws on a CUDA or cuDSS failure.
void SettleAll(Registry& reg) {
    int current = -1;
    cudaGetDevice(&current);
    std::string failure;
    for (auto& entry : reg.slots) {
        Slot& slot = *entry.second;
        Side* side = slot.primary.get();
        if (side == nullptr || !side->pending || !failure.empty()) continue;
        cudaSetDevice(slot.device);
        const cudaStream_t stream = slot.last_stream;
        ffi::Error err = BindSlot(slot, stream);
        if (err.success()) err = SettleSide(reg, slot, *side, stream);
        if (err.success()) err = MarkSlotUse(slot, stream);
        if (!err.success()) failure = err.message();
    }
    if (current >= 0) cudaSetDevice(current);
    if (!failure.empty()) throw std::runtime_error(failure);
}

nb::dict StatsDict() {
    Stats s;
    int64_t live_slots = 0;
    int64_t live_batch_slots = 0;
    int64_t live_sites = 0;
    int64_t live_analyses = 0;
    int64_t live_device_analyses = 0;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(ContractMutex());
        Registry& reg = Reg();
        SettleAll(reg);
        s = reg.stats;
        live_slots = static_cast<int64_t>(reg.slots.size());
        for (const auto& entry : reg.slots) {
            if (entry.second->batch) ++live_batch_slots;
        }
        live_sites = static_cast<int64_t>(reg.sites.size());
        live_analyses = static_cast<int64_t>(reg.analyses.size());
        live_device_analyses = reg.device_analyses;
    }
    nb::dict d;
    d["cold_analyses"] = s.cold_analyses;
    d["seeded_analyses"] = s.seeded_analyses;
    d["factorizations"] = s.factorizations;
    d["refactorizations"] = s.refactorizations;
    d["rebuilds"] = s.rebuilds;
    d["solves"] = s.solves;
    d["transposed_solves"] = s.transposed_solves;
    d["nonfinite"] = s.nonfinite;
    d["singular"] = s.singular;
    d["released_context_errors"] = s.released_context_errors;
    d["host_reads"] = s.host_reads;
    d["zero_fills"] = s.zero_fills;
    d["pinned_blocks_allocated"] = s.pinned_blocks_allocated;
    d["pinned_blocks_reused"] = s.pinned_blocks_reused;
    d["downcast_configurations"] = s.downcast_configurations;
    d["perturbed_pivot_factorizations"] = s.perturbed_pivot_factorizations;
    d["reanalyses"] = s.reanalyses;
    d["static_dispatches"] = s.static_dispatches;
    d["predicate_reads"] = s.predicate_reads;
    d["invalidations"] = s.invalidations;
    d["batched_factorizations"] = s.batched_factorizations;
    d["batched_solves"] = s.batched_solves;
    d["lanes_factored"] = s.lanes_factored;
    d["lanes_factored_unasked"] = s.lanes_factored_unasked;
    d["status_solves"] = s.status_solves;
    d["batch_analyses"] = s.batch_analyses;
    d["lane_rescues"] = s.lane_rescues;
    d["lane_rescue_solves"] = s.lane_rescue_solves;
    d["lanes_factored_alone"] = s.lanes_factored_alone;
    d["lane_solves_alone"] = s.lane_solves_alone;
    d["assemblies"] = s.assemblies;
    d["assembled_lanes"] = s.assembled_lanes;
    d["value_checks"] = s.value_checks;
    d["rebuilds_avoided"] = s.rebuilds_avoided;
    d["storage_value_changes"] = s.storage_value_changes;
    d["storage_asks_unchanged"] = s.storage_asks_unchanged;
    d["deferred_status_reads"] = s.deferred_status_reads;
    d["live_slots"] = live_slots;
    d["live_batch_slots"] = live_batch_slots;
    d["live_sites"] = live_sites;
    d["live_analyses"] = live_analyses;
    d["live_device_analyses"] = live_device_analyses;
    return d;
}

}  // namespace cudss_contract

static void BindContract(nb::module_& m) {
    namespace cc = cudss_contract;
    m.def("contract_analysis_create", &cc::AnalysisCreate, nb::arg("n"), nb::arg("csr_offsets"),
          nb::arg("csr_columns"), nb::arg("nnz"), nb::arg("gather_offsets"),
          nb::arg("gather_entries"), nb::arg("diagonal_add"), nb::arg("reorder_alg"),
          nb::arg("hybrid_mode"), nb::arg("hybrid_device_memory_limit_mb"),
          nb::arg("deterministic"), nb::arg("ir_nsteps"), nb::arg("symmetric_factorization"),
          nb::arg("zero_fill"), nb::arg("downcast_factors"), nb::arg("emulation_mantissa_bits"),
          nb::arg("batch_layout") = 0,
          "Record a sparsity pattern for cuDSS factors held in SolveContexts: the int32 CSR "
          "pattern of the operator (every factor, of either orientation, factors the operator "
          "and solves its transpose in cuDSS's transpose solve mode); the number of coordinate "
          "entries of the topology and the map that assembles coordinate values into the CSR "
          "order (int32 offsets of nse + 1, the int32 coordinate entries grouped by "
          "CSR position with entries in Dirichlet rows stored as -(entry + 1), and a uint8 "
          "marker per position of the Dirichlet diagonals, or an empty array); the reordering "
          "algorithm id (-1 for the library default), hybrid memory mode with its device "
          "limit in MiB (0 for none), deterministic mode (which runs no iterative refinement), "
          "the number of iterative refinement steps, whether factors declared symmetric "
          "use cuDSS's symmetric factorization (otherwise a general LU), whether solves zero "
          "their solution buffers first (0 or 1), cuDSS's downcast mode for float64 factors "
          "(cudssDowncastFactorsMode_t, 0 for none, -1 for the factor's default) and the "
          "mantissa bits of an emulated float64 factorization (0 for none), and how native "
          "batches lay out their lanes (0: one block-diagonal system per tile, the default; "
          "1: cuDSS's uniform batch). Nothing is uploaded "
          "until the first factor. Returns the analysis id.");
    m.def("contract_analysis_set_model_groups", &cc::AnalysisSetModelGroups,
          nb::arg("analysis_id"), nb::arg("mg_devices"), nb::arg("mg_group_size"),
          "Spread every factor of an analysis over a model group with cuDSS MG: the int32 CUDA "
          "ordinals of this process's model groups, mg_group_size (at least 2) consecutive "
          "ordinals per group; each factor spreads over the group that holds the device of its "
          "first call, that device first. Call before the analysis's first factor.");
    m.def("contract_analysis_release", &cc::AnalysisRelease, nb::arg("analysis_id"),
          "Drop the registry's reference to an analysis; sites and slots that use it keep it "
          "alive.");
    m.def("contract_analysis_info", &cc::AnalysisInfo, nb::arg("analysis_id"),
          "Report on an analysis: whether its patterns are on the device, the number of "
          "contexts holding them, the cold and seeded cuDSS analyses run on it (batch tiles "
          "included), refused seeds, the seeded analyses that also took the kept "
          "nested-dissection partition tree, that tree's size in bytes, whether the seed of "
          "sides without matching (symmetric positive definite sides and batch tiles) holds a "
          "tree (tree_seed: None before such a side ran cold, False when cuDSS reported no "
          "tree), the nnz of L + U of the last factorization of a single side "
          "(lu_nnz_forward) and its smallest over largest pivot magnitude "
          "(pivot_ratio_forward; None before one, nan for a non-finite diagonal), the wall "
          "time of the last cold and the last seeded analysis of a single side in seconds, the "
          "device bytes of the uploaded patterns, the hybrid-mode device memory floor (None "
          "outside hybrid mode), n, the number of pattern entries, the cuDSS analyses of batch "
          "tiles, whether the last batch tile analysis was seeded with the kept ordering (None "
          "before one), the wall time of the last batch tile analysis in seconds, and the nnz "
          "of L + U per lane of the last analysed batch tile (lu_nnz_batch; None before one).");
    m.def(
        "contract_set_seeding",
        [](bool enabled) {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(cc::ContractMutex());
            cc::Reg().seeding = enabled;
        },
        nb::arg("enabled"),
        "Seed every later cuDSS analysis of a pattern with the permutation of its first one "
        "(True, the default) or run the reordering for every factor (False).");
    m.def(
        "contract_seeding",
        []() {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(cc::ContractMutex());
            return cc::Reg().seeding;
        },
        "Whether later cuDSS analyses of a pattern are seeded with the permutation of its first "
        "one.");
    m.def(
        "contract_set_batch_tile_lanes",
        [](int64_t lanes) {
            if (lanes < 0) {
                throw std::invalid_argument(
                    "contract_set_batch_tile_lanes takes a non-negative lane count (0 for as "
                    "many lanes as cuDSS's int32 counts allow); got " +
                    std::to_string(lanes) + ".");
            }
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(cc::ContractMutex());
            cc::Reg().batch_tile_lanes = lanes;
        },
        nb::arg("lanes"),
        "Cap the lanes per tile (one stacked system or uniform batch) of the batch slots built "
        "afterwards; 0, "
        "the default, takes as many lanes as keep a tile's row and value counts within int32 "
        "and its factor entries (lanes times the nnz of L + U per lane) within 2**30.");
    m.def(
        "contract_batch_tile_lanes",
        []() {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(cc::ContractMutex());
            return cc::Reg().batch_tile_lanes;
        },
        "The cap on lanes per batch tile set by contract_set_batch_tile_lanes (0 for none).");
    m.def("contract_site_create", &cc::SiteCreate, nb::arg("analysis_id"),
          "Create a factor site on an analysis with an id of its own (a lowered call names its "
          "site by a key computed from the call, and the site is created on the first call "
          "that carries it); its slots are keyed by the context id and lane token (or held "
          "slot) each execution carries. Returns the site id.");
    m.def("contract_storage_declare", &cc::StorageDeclare, nb::arg("analysis_id"),
          nb::arg("context_id"), nb::arg("storage"),
          "Record that a factor of the given context names the shared storage `storage` (the "
          "64-bit key of its storage name, nonzero) on the given analysis. A call on such a "
          "factor whose context id is not a program constant finds the storage without reading "
          "the handle while exactly one open context has declared that key on that analysis.");
    m.def("contract_context_release", &cc::ContextRelease, nb::arg("context_id"),
          "Wait for the devices holding factors of the given SolveContext id, free every slot "
          "and batch slot of that context, drop its references to the device patterns of its "
          "analyses (freed at zero), and record the id as released, so a later call that "
          "carries it fails.");
    m.def("contract_resident_bytes", &cc::ResidentBytes, nb::arg("context_id"),
          "Device bytes held by the slots of the given SolveContext id, batch slots included: "
          "the value copies, the scratch vectors, the single factors of batch lanes (rescues, "
          "and every lane of a batch that runs lane by lane) and the permanent factor storage "
          "cuDSS estimates (CUDSS_DATA_MEMORY_ESTIMATES[0]). The shared device patterns are "
          "reported by contract_analysis_info.");
    m.def("contract_stats", &cc::StatsDict,
          "Process-wide counters of the held-factor handlers, read after every pending factor "
          "verdict is settled (one wait each): cold and seeded cuDSS analyses "
          "(batch tiles included), factorizations, in-place refactorizations and "
          "rebuilds of primary "
          "factors (rebuilds count the numeric phases of batch tiles too), cuDSS solves of "
          "single factors, the solves in transpose mode (transposed_solves: of single factors, "
          "batch tiles and batch lanes), factor requests (batch lanes) whose "
          "values were not "
          "finite or whose factor ended singular, calls refused for a released context, handler calls that "
          "read their handle on the host, solution buffers zeroed before a solve, pinned "
          "host blocks of hybrid-memory factors pinned (pinned_blocks_allocated) and handed "
          "out again from returned blocks (pinned_blocks_reused), single factors and batch tiles "
          "configured to store a float64 factor downcast to float32 (downcast_configurations), "
          "factorizations in which cuDSS perturbed a pivot, sides analysed again because a "
          "factorization of other values than their analysis's ended singular (reanalyses), "
          "handler calls that found their slot from their attributes (static_dispatches), "
          "refactor calls that read their predicate, invalidate calls, numeric phases and "
          "solves of batch tiles (batched_factorizations, batched_solves) with the lanes "
          "they factored (lanes_factored: the lanes their mask asked for), the solves of batch "
          "tiles that settle their lanes' statuses and of symmetric indefinite single "
          "factors that settle theirs (status_solves), cuDSS "
          "analyses of batch tiles, batch lanes factored again alone with matching "
          "(lane_rescues) and solved on that factor (lane_rescue_solves), batch lanes factored "
          "and solved on a single side of their own because the batch runs lane by lane in "
          "deterministic or hybrid memory mode (lanes_factored_alone, lane_solves_alone), "
          "device assemblies "
          "of coordinate values into CSR values (assemblies) with the lanes they covered "
          "(assembled_lanes), the assemblies that also compared the values with the copy they "
          "replaced (value_checks), the rebuilds such a check found unnecessary because the "
          "side already held those values (rebuilds_avoided; batch tiles for a batch), the lanes "
          "a shared-storage refactor factored because the values it keeps differed from the "
          "storage's without being asked (storage_value_changes), the lanes that asked for the "
          "values the storage already held and kept their factor (storage_asks_unchanged), the "
          "refactorization verdicts settled on the host after the refactorization "
          "(deferred_status_reads), and "
          "the live slots, batch slots, sites, "
          "analyses and device-resident analyses.");
    m.def(
        "contract_reset_stats",
        []() {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(cc::ContractMutex());
            cc::SettleAll(cc::Reg());
            cc::Reg().stats = cc::Stats{};
        },
        "Settle every pending factor verdict, then zero the counters reported by "
        "contract_stats(); the live counts are kept.");
    m.def(
        "contract_handlers",
        []() {
            nb::dict d;
            d["diff_soup_cudss_contract_factor_coo_f64"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractFactorF64));
            d["diff_soup_cudss_contract_factor_coo_f32"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractFactorF32));
            d["diff_soup_cudss_contract_solve_coo_f64"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractSolveF64));
            d["diff_soup_cudss_contract_solve_coo_f32"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractSolveF32));
            d["diff_soup_cudss_contract_refactor_coo_f64"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractRefactorF64));
            d["diff_soup_cudss_contract_refactor_coo_f32"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractRefactorF32));
            d["diff_soup_cudss_contract_invalidate_f64"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractInvalidateF64));
            d["diff_soup_cudss_contract_invalidate_f32"] =
                nb::capsule(reinterpret_cast<void*>(kCudssContractInvalidateF32));
            return d;
        },
        "XLA FFI handler capsules of the held-factor handlers by target name, for "
        "jax.ffi.register_ffi_target on the CUDA platform. The factor, solve and refactor "
        "handlers take the values in the coordinate order of the topology (coo) and assemble "
        "them on the device.");
}
