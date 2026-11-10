# Third-party notices: SuperLU_MT

The `_native_superlu` extension in this directory links SuperLU_MT 4.0.2
(https://github.com/xiaoyeli/superlu_mt, commit `1e4bc160e900d601f31410f02ff77ecab59ad275`) and the
reference BLAS routines distributed with it, both statically. SuperLU_MT is
licensed under the BSD 3-clause license reproduced below.

The release wheels' `_native_superlu` is built from commit `1e4bc160e900d601f31410f02ff77ecab59ad275` with `patches/0001-correct-on-several-threads.patch` and `patches/0002-colamd-from-suitesparse.patch` applied.

SuperLU_MT's own COLAMD (`SRC/colamd.c`) is not in the extension: `patches/0002-colamd-from-suitesparse.patch` leaves it out, and the extension uses the SuiteSparse COLAMD that the KLU extension ships (`diff_soup/_internal/klu_ffi/lib/libcolamd.so.3`), whose license is in `diff_soup/_internal/klu_ffi/licenses/COLAMD-License.txt`.

```text
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
(3) Neither the name of Lawrence Berkeley National Laboratory, U.S. Dept. of
Energy nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

The build still compiles against SuperLU_MT's `SRC/colamd.h` for COLAMD's declarations; that header carries this notice:

```text
    Notice:

	Copyright (c) 1998-2007, Timothy A. Davis, All Rights Reserved.

	THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
	EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.

	Permission is hereby granted to use, copy, modify, and/or distribute
	this program, provided that the Copyright, this License, and the
	Availability of the original version is retained on all copies and made
	accessible to the end-user of any code or package that includes COLAMD
	or any modified version of COLAMD.

    Availability:

	The colamd/symamd library is available at http://www.suitesparse.com
	This file is required by the colamd.c, colamdmex.c, and symamdmex.c
	files, and by any C code that calls the routines whose prototypes are
	listed below, or that uses the colamd/symamd definitions listed below.
```
