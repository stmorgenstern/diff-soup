# Third-party notices: SuperLU_MT

The `_native_superlu` extension in this directory links SuperLU_MT 4.0.2
(https://github.com/xiaoyeli/superlu_mt, commit `1e4bc160e900d601f31410f02ff77ecab59ad275`) and the
reference BLAS routines distributed with it, both statically. SuperLU_MT is
licensed under the BSD 3-clause license reproduced below.

The release wheels' `_native_superlu` is built from commit `1e4bc160e900d601f31410f02ff77ecab59ad275` with `patches/0001-correct-on-several-threads.patch` applied.

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

SuperLU_MT's `SRC/colamd.c`:

```text
    Copyright and License:

	Copyright (c) 1998-2007, Timothy A. Davis, All Rights Reserved.
	COLAMD is also available under alternate licenses, contact T. Davis
	for details.

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	This library is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
	Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public
	License along with this library; if not, write to the Free Software
	Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
	USA

	Permission is hereby granted to use or copy this program under the
	terms of the GNU LGPL, provided that the Copyright, this License,
	and the Availability of the original version is retained on all copies.
	User documentation of any code that uses this code or any modified
	version of this code must cite the Copyright, this License, the
	Availability note, and "Used by permission." Permission to modify
	the code and to distribute modified code is granted, provided the
	Copyright, this License, and the Availability note are retained,
	and a notice that the code was modified is included.

    Availability:

	The colamd/symamd library is available at http://www.suitesparse.com
	Appears as ACM Algorithm 836.
```
