// Block Thomas over many independent block-tridiagonal chains (members), threaded over members with OpenMP.
//
// The members are stored back to back, each at its own length: lower, diag, upper (..., S, b, b), right-hand sides
// (..., S, b, m), and offsets (M + 1,) with member k on sites offsets[k] .. offsets[k + 1]. Every axis before the
// last three is a lane axis added by vmap. Block row i of a member is lower_i x_{i-1} + diag_i x_i + upper_i x_{i+1};
// lower at its first site and upper at its last are never read. bad holds one flag per lane.
//
// Factor, without pivoting across grid points: P_i = diag_i - lower_i cp_{i-1}, inv_i = P_i^-1 by an LU with scaled
// row partial pivoting inside the block (the rule of _small_lu.small_lu_factor), cp_i = inv_i upper_i. bad[member]
// holds where a pivot is not finite or at most 16 unit roundoffs of its scale: for b > 1 the largest magnitude of
// its row of P_i (small_lu_factor's rule), for b = 1 |lower_i| + |diag_i| + |upper_i| (the rule of the plain-JAX
// sweep in _member_chain.py, which this kernel must match).
//
// Solve:      r_i = inv_i (c_i - lower_i r_{i-1}),           x_i = r_i - cp_i x_{i+1}.
// Transposed: z_i = c_i - cp_{i-1}^T z_{i-1},                x_i = inv_i^T (z_i - lower_{i+1}^T x_{i+1}).
// Transposes are plain, never conjugate.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

namespace {

constexpr double kSingularPivotRoundoffs = 16.0;

inline bool finite(double v) { return std::isfinite(v); }
inline bool finite(std::complex<double> v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

// inv = a^-1 for one b x b block (a is overwritten). Returns true when the block is singular by the rule above;
// scale1 is the scale of a 1 x 1 pivot.
template <typename T>
bool invert_block(T* a, T* inv, int b, double scale1) {
  constexpr double roundoff = kSingularPivotRoundoffs * std::numeric_limits<double>::epsilon() / 2;
  if (b == 1) {
    inv[0] = T(1.0) / a[0];
    return !finite(a[0]) || std::abs(a[0]) <= roundoff * scale1;
  }
  int perm[16];
  double scale[16];
  for (int i = 0; i < b; ++i) {
    perm[i] = i;
    double s = 0;
    for (int j = 0; j < b; ++j) s = std::max(s, (double)std::abs(a[i * b + j]));
    scale[i] = s > 0 ? s : 1.0;
  }
  bool singular = false;
  for (int k = 0; k < b; ++k) {
    if (k < b - 1) {
      int p = k;
      double best = std::abs(a[k * b + k]) / scale[k];
      for (int i = k + 1; i < b; ++i) {
        double c = std::abs(a[i * b + k]) / scale[i];
        if (c > best) { best = c; p = i; }
      }
      if (p != k) {
        for (int j = 0; j < b; ++j) std::swap(a[k * b + j], a[p * b + j]);
        std::swap(perm[k], perm[p]);
        std::swap(scale[k], scale[p]);
      }
    }
    T pivot = a[k * b + k];
    if (!finite(pivot) || std::abs(pivot) <= roundoff * scale[k]) singular = true;
    for (int i = k + 1; i < b; ++i) {
      T f = a[i * b + k] / pivot;
      for (int j = k + 1; j < b; ++j) a[i * b + j] -= f * a[k * b + j];
      a[i * b + k] = f;
    }
  }
  // Column c of inv solves L U x = P e_c.
  for (int c = 0; c < b; ++c) {
    T y[16];
    for (int i = 0; i < b; ++i) {
      T acc = perm[i] == c ? T(1.0) : T(0.0);
      for (int j = 0; j < i; ++j) acc -= a[i * b + j] * y[j];
      y[i] = acc;
    }
    for (int i = b - 1; i >= 0; --i) {
      T acc = y[i];
      for (int j = i + 1; j < b; ++j) acc -= a[i * b + j] * y[j];
      y[i] = acc / a[i * b + i];
    }
    for (int i = 0; i < b; ++i) inv[i * b + c] = y[i];
  }
  return singular;
}

// c (b x n) = a (b x b) * x (b x n); c -= a * x when Sub.
template <typename T, bool Sub>
inline void gemm(const T* a, const T* x, T* c, int b, int n) {
  for (int i = 0; i < b; ++i)
    for (int j = 0; j < n; ++j) {
      T acc = Sub ? c[i * n + j] : T(0.0);
      for (int k = 0; k < b; ++k) {
        if (Sub) acc -= a[i * b + k] * x[k * n + j];
        else acc += a[i * b + k] * x[k * n + j];
      }
      c[i * n + j] = acc;
    }
}

// The same with a^T.
template <typename T, bool Sub>
inline void gemm_t(const T* a, const T* x, T* c, int b, int n) {
  for (int i = 0; i < b; ++i)
    for (int j = 0; j < n; ++j) {
      T acc = Sub ? c[i * n + j] : T(0.0);
      for (int k = 0; k < b; ++k) {
        if (Sub) acc -= a[k * b + i] * x[k * n + j];
        else acc += a[k * b + i] * x[k * n + j];
      }
      c[i * n + j] = acc;
    }
}

// B > 0 fixes the block size at compile time (the loops unroll); B == 0 takes it from b.
template <typename T, int B>
void factor_member(const T* lo, const T* dg, const T* up, T* inv, T* cp, bool* bad, int64_t n, int b_rt) {
  const int b = B ? B : b_rt;
  const int bb = b * b;
  T piv[256];
  bool singular = false;
  for (int64_t i = 0; i < n; ++i) {
    std::copy(dg + i * bb, dg + (i + 1) * bb, piv);
    if (i > 0) gemm<T, true>(lo + i * bb, cp + (i - 1) * bb, piv, b, b);
    const double scale1 = b == 1 ? std::abs(lo[i]) + std::abs(dg[i]) + std::abs(up[i]) : 0.0;
    singular |= invert_block(piv, inv + i * bb, b, scale1);
    gemm<T, false>(inv + i * bb, up + i * bb, cp + i * bb, b, b);
  }
  *bad = singular;
}

template <typename T, int B>
void solve_member(const T* inv, const T* cp, const T* lo, const T* c, T* x, int64_t n, int b_rt, int m,
                  bool transpose) {
  const int b = B ? B : b_rt;
  const int bb = b * b, bm = b * m;
  std::vector<T> tmp(bm);
  if (!transpose) {
    for (int64_t i = 0; i < n; ++i) {  // x holds r
      std::copy(c + i * bm, c + (i + 1) * bm, tmp.data());
      if (i > 0) gemm<T, true>(lo + i * bb, x + (i - 1) * bm, tmp.data(), b, m);
      gemm<T, false>(inv + i * bb, tmp.data(), x + i * bm, b, m);
    }
    for (int64_t i = n - 2; i >= 0; --i) gemm<T, true>(cp + i * bb, x + (i + 1) * bm, x + i * bm, b, m);
  } else {
    for (int64_t i = 0; i < n; ++i) {  // x holds z
      std::copy(c + i * bm, c + (i + 1) * bm, x + i * bm);
      if (i > 0) gemm_t<T, true>(cp + (i - 1) * bb, x + (i - 1) * bm, x + i * bm, b, m);
    }
    for (int64_t i = n - 1; i >= 0; --i) {
      std::copy(x + i * bm, x + (i + 1) * bm, tmp.data());
      if (i < n - 1) gemm_t<T, true>(lo + (i + 1) * bb, x + (i + 1) * bm, tmp.data(), b, m);
      gemm_t<T, false>(inv + i * bb, tmp.data(), x + i * bm, b, m);
    }
  }
}

// The members of one lane are stored back to back: member k holds sites offsets[k] .. offsets[k + 1] of the flat
// chain. Every axis before the site axis is a lane axis (added by vmap); the offsets of the first lane are used.
struct Members {
  const int32_t* off = nullptr;
  int64_t count = 0, sites = 0;
};

inline bool read_members(ffi::Buffer<ffi::S32> offsets, int64_t sites, Members* out) {
  auto dims = offsets.dimensions();
  if (dims.size() < 1 || dims[dims.size() - 1] < 1) return false;
  out->off = offsets.typed_data();
  out->count = dims[dims.size() - 1] - 1;
  out->sites = out->off[out->count];
  if (out->off[0] != 0 || out->sites != sites) return false;
  for (int64_t k = 0; k < out->count; ++k)
    if (out->off[k + 1] < out->off[k]) return false;
  return true;
}

// Threads pay only when there is enough work to share.
constexpr int64_t kParallelWork = 1 << 15;

template <ffi::DataType DT>
ffi::Error FactorImpl(ffi::Buffer<DT> lower, ffi::Buffer<DT> diag, ffi::Buffer<DT> upper,
                      ffi::Buffer<ffi::S32> offsets, ffi::ResultBuffer<DT> inv, ffi::ResultBuffer<DT> cp,
                      ffi::ResultBuffer<ffi::PRED> bad) {
  using T = ffi::NativeType<DT>;
  auto dims = diag.dimensions();
  if (dims.size() < 3) return ffi::Error::InvalidArgument("member_chain_factor: diag must be (..., S, b, b)");
  const int64_t sites = dims[dims.size() - 3];
  const int b = (int)dims[dims.size() - 1];
  if (b < 1 || b > 16 || dims[dims.size() - 2] != b)
    return ffi::Error::InvalidArgument("member_chain_factor: square blocks of size 1 to 16 are supported");
  if (lower.element_count() != diag.element_count() || upper.element_count() != diag.element_count())
    return ffi::Error::InvalidArgument("member_chain_factor: lower, diag and upper differ in shape");
  Members mem;
  if (!read_members(offsets, sites, &mem))
    return ffi::Error::InvalidArgument("member_chain_factor: offsets do not describe the chain");
  const int64_t bb = b * b, stride = sites * bb;
  const int64_t lanes = stride ? diag.element_count() / stride : 0;
  const T* lo = lower.typed_data();
  const T* dg = diag.typed_data();
  const T* up = upper.typed_data();
  T* iv = inv->typed_data();
  T* c = cp->typed_data();
  bool* bd = bad->typed_data();
  const int64_t jobs = lanes * mem.count;
  std::vector<char> flags(jobs > 0 ? jobs : 1, 0);
  auto run = [&](auto member) {
#pragma omp parallel for schedule(dynamic, 8) if (lanes * stride > kParallelWork)
    for (int64_t job = 0; job < jobs; ++job) {
      const int64_t lane = job / mem.count, k = job % mem.count;
      const int64_t at = lane * stride + mem.off[k] * bb;
      bool singular = false;
      member(lo + at, dg + at, up + at, iv + at, c + at, &singular, (int64_t)(mem.off[k + 1] - mem.off[k]), b);
      flags[job] = singular;
    }
  };
  if (b == 1) run(factor_member<T, 1>);
  else if (b == 2) run(factor_member<T, 2>);
  else run(factor_member<T, 0>);
  for (int64_t lane = 0; lane < lanes; ++lane) {
    bool any = false;
    for (int64_t k = 0; k < mem.count; ++k) any |= (bool)flags[lane * mem.count + k];
    bd[lane] = any;
  }
  return ffi::Error::Success();
}

template <ffi::DataType DT>
ffi::Error SolveImpl(ffi::Buffer<DT> inv, ffi::Buffer<DT> cp, ffi::Buffer<DT> lower, ffi::Buffer<DT> rhs,
                     ffi::Buffer<ffi::S32> offsets, ffi::ResultBuffer<DT> x, bool transpose) {
  using T = ffi::NativeType<DT>;
  auto dims = inv.dimensions();
  auto rdims = rhs.dimensions();
  if (dims.size() < 3 || rdims.size() != dims.size())
    return ffi::Error::InvalidArgument("member_chain_solve: factor (..., S, b, b) and rhs (..., S, b, m) needed");
  const int64_t sites = dims[dims.size() - 3];
  const int b = (int)dims[dims.size() - 1];
  const int m = (int)rdims[rdims.size() - 1];
  if (b < 1 || b > 16 || rdims[rdims.size() - 2] != b || rdims[rdims.size() - 3] != sites)
    return ffi::Error::InvalidArgument("member_chain_solve: rhs does not match the factor");
  Members mem;
  if (!read_members(offsets, sites, &mem))
    return ffi::Error::InvalidArgument("member_chain_solve: offsets do not describe the chain");
  const int64_t bb = b * b, bm = b * m, fs = sites * bb, rs = sites * bm;
  const int64_t lanes = fs ? inv.element_count() / fs : 0;
  if (rs * lanes != rhs.element_count())
    return ffi::Error::InvalidArgument("member_chain_solve: rhs and factor differ in lane count");
  if (rs == 0) return ffi::Error::Success();
  const T* iv = inv.typed_data();
  const T* c = cp.typed_data();
  const T* lo = lower.typed_data();
  const T* r = rhs.typed_data();
  T* out = x->typed_data();
  const int64_t jobs = lanes * mem.count;
  auto run = [&](auto member) {
#pragma omp parallel for schedule(dynamic, 8) if (lanes * rs > kParallelWork)
    for (int64_t job = 0; job < jobs; ++job) {
      const int64_t lane = job / mem.count, k = job % mem.count;
      const int64_t fa = lane * fs + mem.off[k] * bb, ra = lane * rs + mem.off[k] * bm;
      member(iv + fa, c + fa, lo + fa, r + ra, out + ra, (int64_t)(mem.off[k + 1] - mem.off[k]), b, m, transpose);
    }
  };
  if (b == 1) run(solve_member<T, 1>);
  else if (b == 2) run(solve_member<T, 2>);
  else run(solve_member<T, 0>);
  return ffi::Error::Success();
}

}  // namespace

#define DEFINE_PAIR(SUFFIX, DT)                                                                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(kFactor##SUFFIX, FactorImpl<DT>,                                            \
                                ffi::Ffi::Bind()                                                            \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<ffi::S32>>()                                           \
                                    .Ret<ffi::Buffer<DT>>()                                                 \
                                    .Ret<ffi::Buffer<DT>>()                                                 \
                                    .Ret<ffi::Buffer<ffi::PRED>>());                                        \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(kSolve##SUFFIX, SolveImpl<DT>,                                              \
                                ffi::Ffi::Bind()                                                            \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<DT>>()                                                 \
                                    .Arg<ffi::Buffer<ffi::S32>>()                                           \
                                    .Ret<ffi::Buffer<DT>>()                                                 \
                                    .Attr<bool>("transpose"));

DEFINE_PAIR(F64, ffi::F64)
DEFINE_PAIR(C128, ffi::C128)

NB_MODULE(_native_member_chain, m) {
  m.def("registrations", []() {
    nb::dict d;
    d["diffsoup_member_chain_factor_f64"] = nb::capsule(reinterpret_cast<void*>(kFactorF64));
    d["diffsoup_member_chain_solve_f64"] = nb::capsule(reinterpret_cast<void*>(kSolveF64));
    d["diffsoup_member_chain_factor_c128"] = nb::capsule(reinterpret_cast<void*>(kFactorC128));
    d["diffsoup_member_chain_solve_c128"] = nb::capsule(reinterpret_cast<void*>(kSolveC128));
    return d;
  });
}
