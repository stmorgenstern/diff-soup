# Third-party notices: SuiteSparse KLU

The `_native_klu` extension in this directory links the following SuiteSparse
7.10.1 libraries dynamically, as shared libraries that ship next to it (in
`lib/` of an installed wheel; from the vendored tree
`diff_soup/_internal/klu_ffi/vendor/lib/` in a checkout build):

| Library | Version | License |
|---|---|---|
| `libklu.so.2` (KLU) | 2.3.5 | LGPL-2.1-or-later |
| `libbtf.so.2` (BTF) | 2.3.2 | LGPL-2.1-or-later |
| `libamd.so.3` (AMD) | 3.3.3 | BSD-3-Clause |
| `libcolamd.so.3` (COLAMD) | 3.3.4 | BSD-3-Clause |
| `libsuitesparseconfig.so.7` (SuiteSparse_config) | 7.10.1 | BSD-3-Clause |

The libraries are unmodified conda-forge builds of SuiteSparse
(https://github.com/DrTimothyAldenDavis/SuiteSparse). Because KLU and BTF are
linked dynamically, they can be replaced by any compatible build of the same
soname. The GCC runtime libraries they load (`libgomp`, `libgfortran`,
`libquadmath`, `libgcc_s`) are covered by the GCC Runtime Library Exception
(GPL-3.0-or-later; `libquadmath` is LGPL-2.1-or-later); their license texts
are in `diff_soup/_internal/klu_ffi/licenses/`, which also holds the full
GNU LGPL-2.1 text (`libquadmath-COPYING.LIB.txt`) that applies to KLU and BTF.

## KLU (LGPL-2.1-or-later)

```text
KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
Authors: Timothy A. Davis and Ekanathan Palamadai.
SPDX-License-Identifier: LGPL-2.1+

KLU is also available under other licenses; contact authors for details.
http://www.suitesparse.com

--------------------------------------------------------------------------------

KLU is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

KLU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
```

## BTF (LGPL-2.1-or-later)

```text
BTF, Copyright (C) 2004-2013, University of Florida
by Timothy A. Davis and Ekanathan Palamadai.
BTF is also available under other licenses; contact authors for details.
http://www.suitesparse.com

--------------------------------------------------------------------------------

BTF is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

BTF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
```

## AMD (BSD-3-Clause)

```text
AMD, Copyright (c), 1996-2022, Timothy A. Davis,
Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.

Availability:

    http://www.suitesparse.com

-------------------------------------------------------------------------------
AMD License: BSD 3-clause:
-------------------------------------------------------------------------------

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the organizations to which the authors are
          affiliated, nor the names of its contributors may be used to endorse
          or promote products derived from this software without specific prior
          written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.

```

## COLAMD (BSD-3-Clause)

```text
COLAMD, Copyright 1998-2022, Timothy A. Davis.  http://www.suitesparse.com
http://www.suitesparse.com

--------------------------------------------------------------------------------

COLAMD License: BSD 3-clause

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the organizations to which the authors are
          affiliated, nor the names of its contributors may be used to endorse
          or promote products derived from this software without specific prior
          written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
```

## SuiteSparse_config (BSD-3-Clause)

```text
SuiteSparse_config, Copyright (c) 2012-2023, Timothy A. Davis.
All Rights Reserved.
SPDX-License-Identifier: BSD-3-clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the organizations to which the authors are
      affiliated, nor the names of its contributors may be used to endorse
      or promote products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
```
