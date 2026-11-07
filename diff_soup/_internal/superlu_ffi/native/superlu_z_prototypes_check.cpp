// Compile-time check of superlu_z_prototypes.h against SuperLU_MT's own complex128 header: C++
// refuses two declarations of one C function with different parameter types, so a hand-declared
// prototype that differs from slu_mt_zdefs.h fails this translation unit. It defines nothing.
#include "slu_mt_zdefs.h"
#include "superlu_z_prototypes.h"
