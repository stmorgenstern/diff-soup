// The complex128 SuperLU_MT routines superlu_solver.cpp calls, declared by hand: slu_mt_zdefs.h
// cannot share a translation unit with slu_mt_ddefs.h (both define GlobalLU_t), and every other
// type these take is precision-free. superlu_z_prototypes_check.cpp includes this header after
// slu_mt_zdefs.h, so a declaration that differs from the library's fails the build.
#pragma once

#include "slu_dcomplex.h"

extern "C" {
void zCreate_CompCol_Matrix(SuperMatrix*, int_t, int_t, int_t, doublecomplex*, int_t*, int_t*,
                            Stype_t, Dtype_t, Mtype_t);
void zCreate_Dense_Matrix(SuperMatrix*, int_t, int_t, doublecomplex*, int_t, Stype_t, Dtype_t,
                          Mtype_t);
void pzgstrf(superlumt_options_t*, SuperMatrix*, int_t*, SuperMatrix*, SuperMatrix*, Gstat_t*,
             int_t*);
void pzgstrf_init(int_t, fact_t, trans_t, yes_no_t, int_t, int_t, double, yes_no_t, double,
                  int_t*, int_t*, void*, int_t, SuperMatrix*, SuperMatrix*,
                  superlumt_options_t*, Gstat_t*);
void zgstrs(trans_t, SuperMatrix*, SuperMatrix*, int_t*, int_t*, SuperMatrix*, Gstat_t*, int_t*);
}
