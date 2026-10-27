"""Structural content hash for EquationSet.

[content_hash][diff_soup.problem._ir.content_hash.content_hash] returns the
SHA-256 hex digest of an [EquationSet][diff_soup.problem._ir.EquationSet]'s
static spine: its components, globals, constraints, observations, controls,
couplings, scales, residual_scale, families, and declared param/function
namespaces.  Two eqsets hash equal when they lower to the same program; a traced
value (a floating array held by an ``eqx.Module``, a declared parameter value)
contributes its dtype and shape, never its value.

Every node id renders its user callables and static values through one
describer, [_Describer][diff_soup.problem._ir.content_hash._Describer]; the
entry points (``_closure_fingerprint``, ``_stable_value``,
``_eqx_module_fingerprint``, ``_graph_callable_id``,
``_graph_callable_config_id``) only choose its knobs.
"""

from __future__ import annotations

import builtins
import dataclasses
import dis
import hashlib
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
from functools import lru_cache, partial
from types import BuiltinFunctionType, CodeType, FunctionType, MethodType, ModuleType

import equinox as eqx
import jax
import numpy as np

from .constraint import BoundaryIntegralConstraint
from .equation_set import EquationSet
from .observation import BoundaryMeanObservation
from .scales import Scales, identity_scales


def _array_id(value, *, traced: bool = False) -> str:
    """An array's dtype and shape, plus its bytes unless it is a traced value.

    ``traced`` marks a value position: a floating or complex array there is a
    traced value and renders ``dynamic_array:<dtype>:<shape>``.  Every other
    array (and an integer or boolean array anywhere) is a compile-time constant
    and renders its bytes.
    """
    arr = np.asarray(value)
    if traced and np.issubdtype(arr.dtype, np.inexact):
        return f"dynamic_array:{arr.dtype}:{arr.shape}"
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return f"array:{arr.dtype}:{arr.shape}:{digest}"


def _is_variable(value) -> bool:
    return (
        type(value).__name__ == "Variable"
        and hasattr(value, "name")
        and hasattr(value, "shape")
        and hasattr(value, "mass")
    )


def _is_global(value) -> bool:
    return (
        type(value).__name__ == "GlobalVariable"
        and hasattr(value, "name")
        and hasattr(value, "init")
    )


def _build_var_id_to_ref(eqset) -> dict:
    """The owner map that qualifies Variable references within one hash.

    ``"by_id"`` maps each component and family member variable of ``eqset`` to
    ``(owner_name, variable_name)``, ``"by_struct"`` maps ``(name, shape, mass)``
    to the owners of component variables with that structure (for an equal copy
    of a Variable), and ``"by_id_family"`` flags family members.  The map lives
    for one ``content_hash`` call, while ``eqset`` holds every Variable it names.
    """
    by_id: dict[int, tuple[str, str]] = {}
    by_struct: dict[tuple, list[tuple[str, str]]] = {}
    by_id_family: dict[int, bool] = {}
    for comp in eqset.components:
        for v in comp.variables:
            ref = (comp.name, v.name)
            by_id[id(v)] = ref
            by_struct.setdefault((v.name, v.shape, v.mass), []).append(ref)
    for fam in getattr(eqset, "families", ()):
        for v in fam.member_variables:
            by_id[id(v)] = (fam.name, v.name)
            by_id_family[id(v)] = True
    return {"by_id": by_id, "by_struct": by_struct, "by_id_family": by_id_family}


def _qualified_var_name(v, var_id_to_ref) -> str:
    """Owner-qualified name of a Variable / GlobalVariable reference.

    - ``global:{name}`` for a GlobalVariable (globals are unique by name).
    - ``fam:{family}.{name}`` for a family member variable, so it differs from
      a structurally identical component variable.
    - ``{component}.{name}`` for a component variable, found by identity or, for
      an equal copy, by a structure that exactly one component variable has.
    - ``<unowned>.{name}`` otherwise.
    """
    if _is_global(v):
        return f"global:{v.name}"
    if var_id_to_ref:
        ref = var_id_to_ref["by_id"].get(id(v))
        if ref is not None:
            prefix = "fam:" if var_id_to_ref["by_id_family"].get(id(v)) else ""
            return f"{prefix}{ref[0]}.{ref[1]}"
        name = getattr(v, "name", None)
        shape = getattr(v, "shape", None)
        mass = getattr(v, "mass", None)
        if name is not None and shape is not None and mass is not None:
            candidates = var_id_to_ref["by_struct"].get((name, shape, mass), ())
            if len(candidates) == 1:
                return f"{candidates[0][0]}.{candidates[0][1]}"
    return f"<unowned>.{getattr(v, 'name', repr(v))}"


def _ref_id(ref, var_id_to_ref) -> str:
    """A read-set entry: an owner-qualified Variable, or the entry rendered as is."""
    if _is_variable(ref) or _is_global(ref):
        return _qualified_var_name(ref, var_id_to_ref)
    return _stable_value(ref, var_id_to_ref, strict=False)


# Positions a value is rendered at (see ``_Describer``).
_FROZEN, _STRUCTURE, _VALUE = 0, 1, 2

_BUILTINS = vars(builtins)
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})


@lru_cache(maxsize=4096)
def _global_reads(code: CodeType) -> frozenset[tuple[str, str | None]]:
    """The global reads of ``code`` and of every code object nested in its constants.

    A read is ``(name, None)`` for a name the bytecode loads as a global
    (``LOAD_GLOBAL``, ``LOAD_NAME`` in a class body,
    ``LOAD_FROM_DICT_OR_GLOBALS``), and ``(name, attribute)`` as well when
    the load is followed at once by ``LOAD_ATTR``, as ``cfg.D`` and
    ``jnp.where`` compile to; only that first attribute is recorded, so
    ``jnp.linalg.norm`` reads ``jnp.linalg``. An attribute name alone, such as
    the ``exp`` of ``jnp.exp``, is never a read, even when a module global
    shares it. Lambdas, nested functions and class bodies keep their own code
    object among the constants. Memoized per code object: the reads depend on
    the bytecode alone.
    """
    reads: set[tuple[str, str | None]] = set()
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname not in _GLOBAL_LOADS:
            continue
        reads.add((instruction.argval, None))
        following = instructions[index + 1] if index + 1 < len(instructions) else None
        if following is not None and following.opname == "LOAD_ATTR":
            reads.add((instruction.argval, following.argval))
    for const in code.co_consts:
        if isinstance(const, CodeType):
            reads.update(_global_reads(const))
    return frozenset(reads)


def _global_read_error(fn: FunctionType, label: str, value, error: TypeError) -> TypeError:
    """The error for a global ``fn`` reads, named ``label``, that the describer cannot render.

    Names the global and the function that reads it, says what the value is
    (or, for a container, what it holds, read from the describer's own error),
    and states the rule a global must satisfy.
    """
    reader = f"{getattr(fn, '__module__', None)}.{fn.__qualname__}"
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    held = getattr(error, "value_type", None)
    if isinstance(value, jax.core.Tracer):
        what = "a traced value"
    elif held is None or held == kind:
        what = f"a value of type {kind}"
    else:
        what = f"a {kind} holding a value of type {held}"
    result = TypeError(
        f"content_hash cannot describe the module global {label!r} that {reader} reads, "
        f"{what}. A global a function reads must be a plain value (a number, string, array, "
        "tuple, mapping, enum, dataclass, function or eqx.Module), a module or a class, and "
        "so must an attribute it reads from a module or class; hold the value in an "
        "eqx.Module field, or declare it as a parameter or function."
    )
    result.global_read = True
    return result


def _unwrap(fn):
    """Follow ``__wrapped__`` (``functools.wraps``-style wrappers) to the original."""
    seen: set[int] = set()
    while id(fn) not in seen:
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is None:
            break
        seen.add(id(fn))
        fn = wrapped
    return fn


def _class_call(cls):
    for owner in cls.__mro__:
        if "__call__" in owner.__dict__:
            return owner.__dict__["__call__"]
    return None


class _Describer:
    """One walk rendering a value reachable from the IR as a deterministic string.

    Two values render equal only when nothing that decides the lowered program
    differs between them; the rendering never contains an ``id()`` or an
    address, so it is stable across processes.

    Knobs, fixed for the walk:

    - ``strict``: a value the describer cannot render raises ``TypeError``
      (``True``) or renders ``opaque:<module>.<type>`` (``False``).
    - ``held_strict``: the same choice for everything held by an
      ``eqx.Module``, by a bound method's owner, or by a callable object's
      state.  The plain-closure route raises for its own captures and folds
      inside modules, so a recipe whose module holds an undescribable value
      still builds; the graph-callable route raises everywhere.
    - ``trace_modules``: every ``eqx.Module``'s non-static fields are value
      positions (``True``), or only those of a module that itself sits at a
      value position (``False``).
    - ``var_id_to_ref``: the owner map of
      [_qualified_var_name][diff_soup.problem._ir.content_hash._qualified_var_name];
      a Variable renders its owner-qualified name.

    Each value renders at a position:

    - ``_VALUE``: a floating or complex array is a traced value and renders its
      dtype and shape only.
    - ``_STRUCTURE``: every array renders its bytes.  A closure's cells and
      defaults, a static module field, and an object's state are structure;
      a module found there re-opens value positions under ``trace_modules``.
    - ``_FROZEN``: structure that no module below re-opens: a plain dataclass's
      fields, a set's items, a slice, a partial's bound arguments, a mapping
      key, and everything below them.

    What renders, in order: ``None``, numbers, strings, bytes, arrays, numpy
    scalars and dtypes, Variables (owner-qualified), GlobalVariables, containers (tuples,
    lists, mappings, sets), enums, slices, code objects (location-independent),
    Python modules and classes (by name), callables (see
    [callable][diff_soup.problem._ir.content_hash._Describer.callable]),
    plain dataclasses (by field), and objects with a ``shape`` and ``dtype``
    (by both).  A composite that reaches itself renders ``recursive:<name>``.
    """

    __slots__ = ("held_strict", "path", "reader", "strict", "trace_modules", "var_id_to_ref")

    def __init__(self, var_id_to_ref, *, strict: bool, held_strict: bool, trace_modules: bool):
        self.var_id_to_ref = var_id_to_ref
        self.strict = strict
        self.held_strict = strict and held_strict
        self.trace_modules = trace_modules
        self.path: set[int] = set()
        # The module of the function whose global reads are being rendered, or
        # None outside them: a function from another module reached through a
        # global read renders its name and code only.
        self.reader: str | None = None

    def value(self, value, at: int = _STRUCTURE, *, owner: bool = False) -> str:
        """Render ``value`` at position ``at``.

        ``owner`` renders a plain object (one no other rule covers) by its type
        and instance state, as a bound method's owner is, instead of treating
        it as undescribable.
        """
        if value is None or isinstance(value, (bool, int, float, complex, str)):
            return repr(value)
        if isinstance(value, (np.ndarray, jax.Array)):
            return _array_id(value, traced=at == _VALUE)
        if isinstance(value, bytes):
            return f"bytes:{value.hex()}"
        if isinstance(value, np.generic):
            return f"numpy_scalar:{value.dtype}:{value.item()!r}"
        if isinstance(value, np.dtype):
            return f"dtype:{value.str}:{value}"
        if value is Ellipsis:
            return "ellipsis"
        if value is NotImplemented:
            return "not-implemented"
        if _is_variable(value):
            qualified = _qualified_var_name(value, self.var_id_to_ref)
            return f"Variable:{qualified}:{value.shape}:{value.mass}"
        if _is_global(value):
            shape = tuple(np.asarray(value.init).shape)
            role = getattr(value, "mass", "algebraic")
            return f"GlobalVariable:{value.name}:{shape}:mass={role}"
        if isinstance(value, (tuple, list, Mapping, set, frozenset)):
            return self._guarded(value, type(value).__name__, self._container, value, at)
        if isinstance(value, Enum):
            cls = type(value)
            return f"enum:{cls.__module__}.{cls.__qualname__}.{value.name}"
        if isinstance(value, slice):
            bounds = (value.start, value.stop, value.step)
            return "slice:" + ":".join(self.value(item, _FROZEN) for item in bounds)
        if isinstance(value, CodeType):
            return self._code(value)
        if isinstance(value, ModuleType):
            return f"module:{value.__name__}"
        if isinstance(value, type):
            return f"type:{value.__module__}.{value.__qualname__}"
        if isinstance(value, (eqx.Module, FunctionType, BuiltinFunctionType, MethodType, partial)):
            return self.callable(value, at)
        if dataclasses.is_dataclass(value):
            return self._guarded(value, type(value).__qualname__, self._dataclass, value)
        if callable(value):
            return self.callable(value, at)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None and dtype is not None:
            return f"shape_dtype:{np.dtype(dtype)}:{tuple(shape)}"
        if owner:
            return self._guarded(value, type(value).__qualname__, self._object, value, at)
        return self._undescribable(value)

    def callable(self, fn, at: int = _STRUCTURE) -> str:
        """Render a callable by what decides the program it lowers to.

        - A wrapper exposing ``_raw_rhs`` (the residual-form equation adapters)
          renders the callable it wraps, so the hash does not depend on the
          structure the adapter captures at its first evaluation.
        - An ``eqx.Module`` renders its class and every field, by name; static
          fields are structure, the others follow ``trace_modules``.
        - A bound method renders its function and its owner (the owner's
          instance state when it is a plain object).
        - A ``functools.wraps``-style wrapper (``__wrapped__``, as installed by
          a runtime type-check import hook) renders the original.
        - A ``functools.partial`` renders its function and bound arguments.
        - A Python function renders its qualified name, its code object, its
          defaults, its captured cells, and the module globals and builtins
          its code reads: each bare global, and the first attribute it reads
          from a module or class global (``cfg.D``, ``Params.D``,
          ``jnp.where``). A global renders as a captured cell holding the same
          object does, so rebinding a global or attribute the function reads
          changes the rendering; a module or class read without an attribute
          renders by name. A function from another module reached through
          such a read renders its qualified name and code object only, never
          its own globals or closure; a function from the reading function's
          module renders in full.
        - A builtin renders its qualified name.
        - Any other callable object renders its class, ``__call__``, and
          instance state (``__dict__`` and ``__slots__``); its state is fixed
          after construction, or two builds of one recipe would differ.
        """
        raw = getattr(fn, "_raw_rhs", None)
        if raw is not None:
            return self.callable(raw, at)
        if isinstance(fn, eqx.Module):
            return self._held(self._guarded, fn, type(fn).__qualname__, self._module, fn, at)
        if isinstance(fn, MethodType):
            inner = _FROZEN if at == _FROZEN else _STRUCTURE
            func = self.callable(fn.__func__, inner)
            owner = self._held(self.value, fn.__self__, inner, owner=True)
            return f"method:{func}|self={owner}"
        original = _unwrap(fn)
        if original is not fn:
            return self.callable(original, at)
        if isinstance(fn, partial):
            keywords = {} if fn.keywords is None else fn.keywords
            return "|".join(
                (
                    "partial:" + self.callable(fn.func, _FROZEN),
                    "args=" + self.value(fn.args, _FROZEN),
                    "keywords=" + self.value(keywords, _FROZEN),
                )
            )
        module = getattr(fn, "__module__", type(fn).__module__)
        qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "")
        if isinstance(fn, FunctionType):
            name = f"{module}.{qualname}"
            if self.reader is not None and module != self.reader:
                return f"function:{name}|" + self._code(fn.__code__)
            return self._guarded(fn, name, self._function, fn, name, at)
        if isinstance(fn, BuiltinFunctionType):
            return f"builtin:{module}.{qualname}"
        return self._held(self._guarded, fn, type(fn).__qualname__, self._object, fn, at)

    def _held(self, render, *args, **kwargs) -> str:
        strict = self.strict
        self.strict = self.held_strict
        try:
            return render(*args, **kwargs)
        finally:
            self.strict = strict

    def _guarded(self, value, name: str, render, *args) -> str:
        key = id(value)
        if key in self.path:
            return f"recursive:{name}"
        self.path.add(key)
        try:
            return render(*args)
        finally:
            self.path.discard(key)

    def _container(self, value, at: int) -> str:
        kind = type(value).__name__
        if isinstance(value, (tuple, list)):
            return f"{kind}:[" + ",".join(self.value(item, at) for item in value) + "]"
        if isinstance(value, Mapping):
            items = sorted(
                (self.value(key, _FROZEN), self.value(item, at)) for key, item in value.items()
            )
            return f"{kind}:{{" + ",".join(f"{key}:{item}" for key, item in items) + "}"
        items = sorted(self.value(item, _FROZEN) for item in value)
        return f"{kind}:{{" + ",".join(items) + "}"

    def _module(self, mod, at: int) -> str:
        cls = type(mod)
        parts = [f"eqx:{cls.__module__}.{cls.__qualname__}"]
        for field in sorted(dataclasses.fields(mod), key=lambda item: item.name):
            if at == _FROZEN:
                field_at = _FROZEN
            elif field.metadata.get("static", False):
                field_at = _STRUCTURE
            elif self.trace_modules or at == _VALUE:
                field_at = _VALUE
            else:
                field_at = _STRUCTURE
            parts.append(f"{field.name}=" + self.value(getattr(mod, field.name, None), field_at))
        return "|".join(parts)

    def _dataclass(self, value) -> str:
        cls = type(value)
        parts = [f"dataclass:{cls.__module__}.{cls.__qualname__}"]
        for field in sorted(dataclasses.fields(value), key=lambda item: item.name):
            parts.append(f"{field.name}=" + self.value(getattr(value, field.name, None), _FROZEN))
        return "|".join(parts)

    def _function(self, fn: FunctionType, name: str, at: int) -> str:
        inner = _FROZEN if at == _FROZEN else _STRUCTURE
        # Captured values and defaults render in full whatever module they
        # come from: the library's own closures hold user callbacks there.
        reader = self.reader
        self.reader = None
        try:
            cells = []
            for cell in fn.__closure__ or ():
                try:
                    contents = cell.cell_contents
                except ValueError:
                    cells.append("empty-cell")
                else:
                    cells.append(self.value(contents, inner))
            parts = (
                f"function:{name}",
                self._code(fn.__code__),
                "defaults=" + self.value(fn.__defaults__ or (), inner),
                "kwdefaults=" + self.value(fn.__kwdefaults__ or {}, inner),
                "closure=[" + ",".join(cells) + "]",
            )
        finally:
            self.reader = reader
        return "|".join(parts + ("globals=[" + ",".join(self._globals(fn, inner)) + "]",))

    def _globals(self, fn: FunctionType, at: int) -> list[str]:
        """``name=<rendering>`` for each module global and builtin ``fn`` reads, sorted.

        Each name its bytecode loads as a global resolves in ``fn.__globals__``,
        then in ``builtins``; a name found in neither is unbound and is skipped.
        An attribute read from a module or class global (``cfg.D``) resolves on
        that object, and is skipped when the attribute is missing. A global
        renders as a closure cell holding it does, at ``at``, with ``fn``'s
        module as the reader; a builtin renders its qualified name.
        """
        namespace = fn.__globals__
        reader = self.reader
        self.reader = getattr(fn, "__module__", None)
        try:
            reads = []
            for name, attribute in sorted(
                _global_reads(fn.__code__), key=lambda read: (read[0], read[1] or "")
            ):
                if name not in namespace:
                    if attribute is None and name in _BUILTINS:
                        reads.append(f"{name}=builtins.{name}")
                    continue
                value = namespace[name]
                if attribute is None:
                    reads.append(f"{name}=" + self._global(fn, name, value, at))
                elif isinstance(value, (ModuleType, type)):
                    try:
                        member = getattr(value, attribute)
                    except AttributeError:
                        continue
                    label = f"{name}.{attribute}"
                    reads.append(f"{label}=" + self._global(fn, label, member, at))
            return reads
        finally:
            self.reader = reader

    def _global(self, fn: FunctionType, label: str, value, at: int) -> str:
        """Render the global ``label`` that ``fn`` reads, naming both when it cannot."""
        try:
            return self.value(value, at)
        except TypeError as error:
            if getattr(error, "global_read", False):
                raise
            raise _global_read_error(fn, label, value, error) from None

    def _code(self, code: CodeType) -> str:
        """A code object's bytecode, constants, names and signature, not its location."""
        return _code_id(code)

    def _object(self, obj, at: int) -> str:
        cls = type(obj)
        call = _class_call(cls)
        if isinstance(call, FunctionType):
            call_id = self.callable(call)
        elif call is None:
            call_id = "none"
        else:
            call_name = getattr(call, "__qualname__", getattr(call, "__name__", ""))
            call_id = f"{type(call).__module__}.{type(call).__qualname__}:{call_name}"
        state: dict[str, object] = {}
        try:
            state.update(vars(obj))
        except TypeError:
            pass
        for owner in cls.__mro__:
            slots = owner.__dict__.get("__slots__", ())
            for slot in (slots,) if isinstance(slots, str) else slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                attr = slot
                if slot.startswith("__") and not slot.endswith("__"):
                    attr = f"_{owner.__name__.lstrip('_')}{slot}"
                if hasattr(obj, attr):
                    state[f"{owner.__module__}.{owner.__qualname__}.{slot}"] = getattr(obj, attr)
        identity = (
            getattr(obj, "__module__", None),
            getattr(obj, "__qualname__", None),
            getattr(obj, "__name__", None),
        )
        inner = _FROZEN if at == _FROZEN else _STRUCTURE
        return "|".join(
            (
                f"object:{cls.__module__}.{cls.__qualname__}",
                f"identity={identity!r}",
                f"call={call_id}",
                "state=" + self.value(state, inner),
            )
        )

    def _undescribable(self, value) -> str:
        cls = type(value)
        name = f"{cls.__module__}.{cls.__qualname__}"
        if self.strict:
            error = TypeError(
                f"content_hash cannot describe a captured value of type {name}. Capture "
                "only deterministic plain values (numbers, strings, arrays, tuples, "
                "mappings, enums, dataclasses, functions, eqx.Modules), hold the value "
                "in an eqx.Module field, or declare it as a parameter or function."
            )
            error.value_type = name
            raise error
        return f"opaque:{name}"


@lru_cache(maxsize=1024)
def _code_id(code: CodeType) -> str:
    """A code object's bytecode, constants, names and signature, not its location.

    Memoized per code object: its constants are plain values (numbers,
    strings, tuples, nested code objects), so the rendering depends on nothing
    else, and a function referenced from many places renders once.
    """
    describer = _Describer(None, strict=True, held_strict=True, trace_modules=False)
    return "|".join(
        (
            f"code={code.co_code.hex()}",
            "consts=" + ",".join(describer.value(item, _FROZEN) for item in code.co_consts),
            f"names={code.co_names!r}",
            f"varnames={code.co_varnames!r}",
            f"freevars={code.co_freevars!r}",
            f"cellvars={code.co_cellvars!r}",
            f"args={code.co_argcount}:{code.co_posonlyargcount}:{code.co_kwonlyargcount}",
            f"flags={code.co_flags}",
            f"exception_table={code.co_exceptiontable.hex()}",
        )
    )


def _closure_describer(var_id_to_ref, strict: bool) -> _Describer:
    return _Describer(var_id_to_ref, strict=strict, held_strict=False, trace_modules=True)


def _graph_describer(var_id_to_ref) -> _Describer:
    return _Describer(var_id_to_ref, strict=True, held_strict=True, trace_modules=False)


def _stable_value(value, var_id_to_ref=None, *, strict: bool = True) -> str:
    """Render a static value reachable from the IR (a cell, a default, a field).

    The plain-closure rules: ``strict=True`` raises ``TypeError`` for a value
    the describer cannot render, outside ``eqx.Module`` contents and bound-method
    owners, which fold to ``opaque:<type>``; ``strict=False`` never raises.
    Every ``eqx.Module`` contributes its floating arrays' dtype and shape only.
    """
    return _closure_describer(var_id_to_ref, strict).value(value)


def _closure_fingerprint(fn, var_id_to_ref=None, *, strict: bool = True) -> str:
    """Render a user callable by what decides its lowered residual.

    The plain-closure rules of
    [_stable_value][diff_soup.problem._ir.content_hash._stable_value]; see
    [_Describer.callable][diff_soup.problem._ir.content_hash._Describer.callable]
    for what each kind of callable contributes.
    """
    return _closure_describer(var_id_to_ref, strict).callable(fn)


def _eqx_module_fingerprint(mod: eqx.Module, var_id_to_ref=None) -> str:
    """Render an ``eqx.Module`` by its structure, never by its floating leaves.

    A floating array in a non-static field is a traced value: a new value must
    not change the hash, or every optimizer step would rebuild the lowered
    problem; it contributes its dtype and shape.  Everything else the module
    holds is structure, folded field by field without raising (an
    undescribable value renders ``opaque:<type>``): static fields, integer and
    boolean arrays (bytes included), nested modules, and every callable,
    including a user closure wrapped by the traced adapter.
    """
    return _closure_describer(var_id_to_ref, strict=False).callable(mod)


def _graph_callable_config_id(value, var_id_to_ref=None, *, force_static: bool = False) -> str:
    """Render a value under the graph-callable rules, raising for anything opaque.

    A floating array at a value position (the top level, unless
    ``force_static``, and an ``eqx.Module``'s non-static fields reached through
    value positions) contributes its dtype and shape: those are the leaves
    ``eqx.is_inexact_array`` exposes as parameters.  Every other array (static
    fields, closure captures, dataclass fields, and everything under
    ``force_static``) contributes its bytes.  A value the describer cannot
    render raises ``TypeError``.
    """
    return _graph_describer(var_id_to_ref).value(value, _STRUCTURE if force_static else _VALUE)


def _graph_callable_id(fn, var_id_to_ref=None) -> str:
    """Render a graph callback (weight function, exchange) under the graph-callable rules."""
    return _graph_describer(var_id_to_ref).callable(fn, _VALUE)


def _param_shape_dtype_id(value) -> str:
    """A declared value leaf: dtype and shape, plus the bytes of a constant.

    A floating or complex array (a ``member_params`` entry, a numeric graph
    weight, a declared value) is a traced value: only its dtype and shape enter
    the hash, so a new value keeps the structure while a leading-axis or
    per-member shape change does not.  Any other leaf (an integer or boolean
    array, a Python number) is a compile-time constant and contributes its
    bytes.
    """
    if eqx.is_inexact_array(value):
        return f"{np.dtype(value.dtype)}:{tuple(value.shape)}"
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        return f"{arr.dtype}:{arr.shape}"
    return _array_id(arr)


_MAX_DOMAIN_ID_CACHE_ENTRIES = 256
_DOMAIN_ID_CACHE: OrderedDict[int, tuple[weakref.ref, str]] = OrderedDict()


def _domain_id(domain) -> str:
    """A domain's structural id: its ``identifier``, or its kind, sizes and geometry.

    Memoized per live domain object.  Domains are ``eqx.Module`` values holding
    arrays, so they are unhashable and cannot key a ``WeakKeyDictionary``; the
    memo is keyed by ``id`` and guarded by a weak reference, so an entry whose
    domain has died is a miss.
    """
    key = id(domain)
    cached = _DOMAIN_ID_CACHE.get(key)
    if cached is not None and cached[0]() is domain:
        _DOMAIN_ID_CACHE.move_to_end(key)
        return cached[1]
    value = _domain_id_uncached(domain)
    try:
        _DOMAIN_ID_CACHE[key] = (weakref.ref(domain), value)
    except TypeError:
        return value
    _DOMAIN_ID_CACHE.move_to_end(key)
    while len(_DOMAIN_ID_CACHE) > _MAX_DOMAIN_ID_CACHE_ENTRIES:
        _DOMAIN_ID_CACHE.popitem(last=False)
    return value


# Domain attributes folded by value (scalars) and by bytes (geometry arrays).
_DOMAIN_SCALARS = "dim discretization_type ele_type coord_system _method periodic_axes".split()
_DOMAIN_ARRAYS = (
    "points cells nodes faces cell_widths node_distances cell_volumes face_areas "
    "metric_gradient_correction lhopital_extra_d2"
).split()


def _domain_id_uncached(domain) -> str:
    ident = getattr(domain, "identifier", None)
    if ident is not None:
        return f"{type(domain).__name__}:{ident}"
    parts = [type(domain).__name__, f"n={getattr(domain, 'num_nodes', '?')}"]
    seen = set()
    for attr in _DOMAIN_SCALARS:
        if hasattr(domain, attr):
            seen.add(attr)
            parts.append(f"{attr}={getattr(domain, attr)}")
    for attr in _DOMAIN_ARRAYS:
        if hasattr(domain, attr):
            seen.add(attr)
            parts.append(f"{attr}={_array_id(getattr(domain, attr))}")
    if dataclasses.is_dataclass(domain):
        for field in dataclasses.fields(domain):
            if field.name in seen:
                continue
            value = getattr(domain, field.name)
            if isinstance(value, (np.ndarray, jax.Array)):
                parts.append(f"{field.name}={_array_id(value)}")
                continue
            try:
                parts.append(f"{field.name}={_stable_value(value)}")
            except TypeError:
                parts.append(f"{field.name}=opaque:{type(value).__name__}")
    return ":".join(parts)


def _selector_fold(selector) -> str:
    """Fold a [BoundarySet][diff_soup.problem._ir.BoundarySet] selector into a canonical string.

    Different selections differ and equal ones collide: tag names are sorted,
    so two spellings of one set fold identically.
    """
    return selector.reduce(
        on_tags=lambda names: "tags(" + ",".join(sorted(names)) + ")",
        on_nodes=lambda idx: "nodes(" + ",".join(map(str, idx)) + ")",
        on_union=lambda a, b: "U(" + a + "," + b + ")",
        on_difference=lambda a, b: "D(" + a + "," + b + ")",
    )


def _fold_callables(node, attrs, var_id_to_ref) -> str:
    """``:attr=<fingerprint>`` for each attribute of ``attrs`` that ``node`` holds."""
    out = []
    for attr in attrs:
        fn = getattr(node, attr, None)
        if fn is not None:
            out.append(f":{attr}={_closure_fingerprint(fn, var_id_to_ref)}")
    return "".join(out)


_BC_CALLABLES = ("value_fn", "flux_fn", "alpha_fn", "beta_fn")


def _bc_id(bc, var_id_to_ref=None) -> str:
    """A boundary condition: its kind, variable, selection, options and callables.

    The pinned component of a Dirichlet condition, a flux condition's handle
    (a ``flux_total`` constraint links to it) and its cross-facet locality flag,
    and a periodic condition's axes are structural.
    """
    bs = bc.where
    parts = [f"{type(bc).__name__}:{_qualified_var_name(bc.var, var_id_to_ref)}"]
    parts.append(_selector_fold(bs.selector))
    for attr in ("component", "bc_name"):
        value = getattr(bc, attr, None)
        if value is not None:
            parts.append(f"{attr}={value!r}")
    if getattr(bc, "cross_facet", False):
        parts.append("cross_facet")
    axes = getattr(bc, "axes", None)
    if axes is not None:
        parts.append(f"axes={tuple(axes)!r}")
    return ":".join(parts) + _fold_callables(bc, _BC_CALLABLES, var_id_to_ref)


def _eq_id(eq, var_id_to_ref=None) -> str:
    """An equation: its name, owner-qualified writes and reads, and its rhs."""
    writes = getattr(eq, "vars", None)
    writes = (eq.var,) if writes is None else writes
    return (
        f"{type(eq).__name__}:{eq.name}"
        f":writes={','.join(_qualified_var_name(v, var_id_to_ref) for v in writes)}"
        f":reads={','.join(_qualified_var_name(x, var_id_to_ref) for x in eq.inputs)}"
        f":fn={_closure_fingerprint(eq.rhs_fn, var_id_to_ref)}"
    )


_CONSTRAINT_CALLABLES = (
    "rhs_fn",
    "current_fn",
    "voltage_fn",
    "integrand_fn",
    "reduce_fn",
    "target_fn",
)


def _constraint_id(cs, var_id_to_ref=None) -> str:
    """A global constraint: its kind, closed global, reads, options and callables.

    An operational table contributes its modes and seed threshold, never its
    runtime mode/target/seed values (solve data); a current read from a field's
    residual rows is identified by that field.
    """
    parts = [f"Constraint:{type(cs).__name__}:{cs.name}:closes={cs.closes.name}"]
    inputs = ",".join(_ref_id(x, var_id_to_ref) for x in getattr(cs, "inputs", ()))
    if inputs:
        parts.append(f"inputs=[{inputs}]")
    field = getattr(cs, "field", None)
    if field is not None:
        parts.append(f"field={_qualified_var_name(field, var_id_to_ref)}")
    if isinstance(cs, BoundaryIntegralConstraint):
        where = _selector_fold(cs.selector) if cs.selector is not None else f"tag={cs.tag}"
        parts.append(f"where={where}")
    for attr in ("reads", "from_bc", "modes"):
        names = getattr(cs, attr, None)
        if names is not None:
            parts.append(f"{attr}=[{','.join(names)}]")
    threshold = getattr(cs, "seed_voltage_threshold", None)
    if threshold is not None:
        parts.append(f"seed_voltage_threshold={float(threshold)!r}")
    scale = float(getattr(cs, "integrand_scale", 1.0))
    if scale != 1.0:
        parts.append(f"integrand_scale={scale!r}")
    attrs = _CONSTRAINT_CALLABLES
    if field is not None:
        attrs = tuple(a for a in attrs if a != "current_fn")
    return ":".join(parts) + _fold_callables(cs, attrs, var_id_to_ref)


def _observation_id(obs, var_id_to_ref=None) -> str:
    """An observation: its kind, name, boundary tag, read set and callable."""
    base = f"Observation:{type(obs).__name__}:{obs.name}"
    if isinstance(obs, BoundaryMeanObservation):
        base = f"{base}:tag={obs.tag}"
    inputs = ",".join(_ref_id(x, var_id_to_ref) for x in getattr(obs, "inputs", ()))
    if inputs:
        base = f"{base}:inputs=[{inputs}]"
    return base + _fold_callables(obs, ("fn",), var_id_to_ref)


def _control_id(ctl, var_id_to_ref=None) -> str:
    """A control: its kind, name, the global it drives (by name) and its callable."""
    base = f"Control:{type(ctl).__name__}:{ctl.name}:binds_to={getattr(ctl, 'binds_to', None)}"
    return base + _fold_callables(ctl, ("fn",), var_id_to_ref)


def _coupling_id(cp, var_id_to_ref=None) -> str:
    """A coupling: owner-qualified source and target, its TransferOp, and its mode.

    A read-through coupling lowers to a view injection with no state slot or
    residual row, a structurally distinct build from the DOF-overwrite path;
    its target alias, when not a component variable, renders
    ``<unowned>.<name>``.
    """
    parts = [
        cp.name,
        f"src={_qualified_var_name(cp.source, var_id_to_ref)}",
        f"tgt={_qualified_var_name(cp.target, var_id_to_ref)}",
        f"op={_eqx_module_fingerprint(cp.op, var_id_to_ref)}",
    ]
    if getattr(cp, "read_through", False):
        parts.append("read_through")
    if getattr(cp, "target_owner", None) is not None:
        parts.append(f"target_owner={cp.target_owner!r}")
    return ":".join(parts)


def _scales_id(scales: Scales) -> str:
    """Structural id of non-identity Scales (their values are static config)."""
    return (
        "Scales:"
        f"state_ref={tuple(sorted(scales.state_ref.items()))}:"
        f"state_offset={tuple(sorted(scales.state_offset.items()))}:"
        f"time_ref={float(scales.time_ref)!r}:"
        f"integrand_ref={tuple(sorted(scales.integrand_ref.items()))}"
    )


def _residual_scale_id(residual_scale) -> str:
    """Structural id of a non-empty residual_scale mapping.

    The per-row scale factors are static config whose values decide the
    residual-row scaling, so they enter the hash; keys are
    ``(owner_name, var_name)`` tuples, sorted.
    """
    items = sorted((repr(k), repr(float(v))) for k, v in residual_scale.items())
    return "ResidualScale:" + ",".join(f"{k}={v}" for k, v in items)


def _family_graph_id(graph, var_id_to_ref=None) -> str:
    """A family graph's structure: edges, direction, and weight function or weight shape."""
    edges = tuple(zip(graph.edge_sources, graph.edge_targets))
    parts = [
        f"graph:{graph.name}",
        f"member_count={int(graph.member_count)}",
        f"directed={bool(graph.directed)!r}",
        f"edges={edges!r}",
    ]
    if callable(graph.weights):
        parts.append("weight_fn=" + _graph_callable_id(graph.weights, var_id_to_ref))
    else:
        parts.append("weights=" + _param_shape_dtype_id(graph.weights))
    return ":".join(parts)


def _family_graph_exchange_id(exchange, var_id_to_ref=None) -> str:
    """A declared conservative exchange: graph, writes, reads and callable."""
    return ":".join(
        (
            f"graph_exchange:{exchange.name}",
            f"graph={exchange.graph}",
            f"writes={exchange.writes}",
            f"reads={exchange.reads!r}",
            f"weight_reads={exchange.weight_reads!r}",
            "fn=" + _graph_callable_id(exchange.fn, var_id_to_ref),
        )
    )


def _family_id(fam, var_id_to_ref=None) -> str:
    """A [Family][diff_soup.problem._ir.Family]'s static spine.

    Folds the name, count, population offset (a ragged bucket's offset shifts
    every member's ``m.member_index``), declared ``stencil_radius``, member
    domain, owner-qualified member variables (``fam:{family}.{var}``), member
    equations, BCs and couplings, graphs and exchanges, ``scale`` and
    ``residual_scale`` values, and the operator factory.  ``member_params``
    contribute their dtypes and shapes only: their values are traced leaves.
    """
    parts = [f"Family:{fam.name}:count={fam.count}"]
    offset = int(getattr(fam, "member_index_offset", 0))
    if offset:
        parts.append(f"member_index_offset={offset}")
    if getattr(fam, "stencil_radius", None) is not None:
        parts.append(f"stencil_radius={int(fam.stencil_radius)}")
    parts.append(f"domain={_domain_id(fam.domain)}")
    for v in fam.member_variables:
        qualified = _qualified_var_name(v, var_id_to_ref)
        parts.append(
            f"member_var:{qualified}:{getattr(v, 'shape', None)}:{getattr(v, 'mass', None)}"
        )
    parts.extend(f"member_eq:{_eq_id(eq, var_id_to_ref)}" for eq in fam.member_equations)
    parts.extend(f"member_bc:{_bc_id(bc, var_id_to_ref)}" for bc in fam.member_bcs)
    parts.extend(
        f"member_coupling:{_coupling_id(cp, var_id_to_ref)}" for cp in fam.member_couplings
    )
    for key in sorted(fam.member_params):
        parts.append(f"member_param:{key}:{_param_shape_dtype_id(fam.member_params[key])}")
    parts.extend(_family_graph_id(graph, var_id_to_ref) for graph in getattr(fam, "graphs", ()))
    parts.extend(
        _family_graph_exchange_id(exchange, var_id_to_ref)
        for exchange in getattr(fam, "graph_exchanges", ())
    )
    if fam.scale is not None:
        parts.append(f"scale={_stable_value(fam.scale, var_id_to_ref)}")
    if fam.residual_scale is not None:
        parts.append(f"residual_scale={_stable_value(fam.residual_scale, var_id_to_ref)}")
    if fam.operator_factory is not None:
        parts.append("operator_factory=" + _stable_value(fam.operator_factory))
    return "|".join(parts)


def iter_declared_closures(container):
    """Yield ``(dotted_path, closure)`` for every user closure on a container.

    An [EquationSet][diff_soup.problem._ir.EquationSet] stores its equations under ``components`` (a static
    field, so a pytree walk misses them); a ``LoweredProblem`` stores them under
    ``_equations`` / ``_constraints`` / ... .  Both are walked structurally here
    so the declared param/function namespaces are reachable from either.  The
    dotted path locates the closure inside ``container`` (attribute names, with
    integer components indexing a tuple), so a caller can name the leaves it
    reaches.
    """
    comps = getattr(container, "components", None)
    if comps is not None:
        for i, comp in enumerate(comps):
            for j, eq in enumerate(comp.equations):
                yield f"components.{i}.equations.{j}.rhs_fn", getattr(eq, "rhs_fn", None)
            for j, bc in enumerate(comp.bcs):
                for attr in ("value_fn", "flux_fn", "alpha_fn", "beta_fn"):
                    yield f"components.{i}.bcs.{j}.{attr}", getattr(bc, attr, None)
        constraints = ("constraints", getattr(container, "constraints", ()))
        observations = ("observations", getattr(container, "observations", ()))
        controls = ("controls", getattr(container, "controls", ()))
    else:
        for i, eq in enumerate(getattr(container, "_equations", ())):
            yield f"_equations.{i}.rhs_fn", getattr(eq, "rhs_fn", None)
        constraints = ("_constraints", getattr(container, "_constraints", ()))
        observations = ("_observations", getattr(container, "_observations", ()))
        controls = ("_controls", getattr(container, "_controls", ()))
    base, items = constraints
    for i, cs in enumerate(items):
        # ``rhs_fn`` reaches a differential global's equation and ``voltage_fn``
        # the operational table's terminal-voltage closure, so a Recipe whose
        # only closures live on global rows still exposes its namespaces.
        attrs = ("current_fn", "voltage_fn", "integrand_fn", "reduce_fn", "rhs_fn", "target_fn")
        for attr in attrs:
            yield f"{base}.{i}.{attr}", getattr(cs, attr, None)
    for base, items in (observations, controls):
        for i, node in enumerate(items):
            yield f"{base}.{i}.fn", getattr(node, "fn", None)


def _named_values_from_defaults(fn):
    """Extract the ``(params, functions)`` namespaces closed over by a plain adapter.

    A non-learnable, non-module closure is wrapped in a plain function whose
    default arguments carry the two named-value namespaces positionally
    (``params`` before ``functions``).  Returns ``(None, None)`` when ``fn`` is
    not such a closure.  Callers reach the wrapped layers through
    [declared_callable_layers][diff_soup.problem._ir.content_hash.declared_callable_layers], so this inspects the layer it is
    handed and no further.
    """
    if fn is None:
        return None, None
    defaults = getattr(fn, "__defaults__", None)
    if not defaults:
        return None, None
    nvs = [d for d in defaults if type(d).__name__ == "_NamedValues"]
    if len(nvs) >= 2:
        return nvs[0], nvs[1]
    return None, None


def declared_callable_layers(fn):
    """Yield ``(dotted_suffix, layer)`` down to the callable that owns namespaces.

    A lowered closure is wrapped: the residual-form adapter, the bundled-return
    guard, and the scaled Dirichlet value all hold the callable below them in
    ``inner``, and a plain wrapped function holds it in ``__wrapped__``.  The
    suffix is the dotted path from ``fn`` to that layer (``""`` for ``fn``
    itself), so a caller can address the layer it stops on.
    """
    seen: set[int] = set()
    suffix = ""
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        yield suffix, fn
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            fn = wrapped
            suffix = f"{suffix}.__wrapped__" if suffix else "__wrapped__"
            continue
        fn = getattr(fn, "inner", None)
        suffix = f"{suffix}.inner" if suffix else "inner"


def declared_named_values(container):
    """Return the shared ``(params, functions)`` namespaces declared on ``container``.

    ``container`` is an [EquationSet][diff_soup.problem._ir.EquationSet] or a ``LoweredProblem``.  Every
    closure shares one params namespace and one functions namespace, so a single
    representative is returned (``None`` for a namespace that is absent).
    """
    for _path, fn in iter_declared_closures(container):
        for _suffix, layer in declared_callable_layers(fn):
            if type(layer).__name__ == "_TracedQAdapter":
                return getattr(layer, "params", None), getattr(layer, "functions", None)
            params, functions = _named_values_from_defaults(layer)
            if params is not None or functions is not None:
                return params, functions
    return None, None


def _declared_leaf_fold(value) -> str:
    """Fold every array or number leaf of a declared value.

    Follows [_param_shape_dtype_id][diff_soup.problem._ir.content_hash._param_shape_dtype_id]:
    a floating leaf folds its shape and dtype, so a new value of a fixed-shape
    leaf keeps the structure while a shape change (a degree-4 vs degree-2
    expansion, a longer param vector) does not; an integer or boolean leaf,
    or a Python number, is a compile-time constant and folds its value.
    Returns ``"()"`` for a value with no such leaves (e.g. a non-module
    callable).
    """
    fold = []
    for leaf in jax.tree_util.tree_leaves(value):
        if isinstance(leaf, (np.ndarray, jax.Array, bool, int, float)):
            fold.append(_param_shape_dtype_id(leaf))
    return ",".join(fold) if fold else "()"


def _declared_param_function_id(eqset) -> str:
    """Declared params/functions by name and leaf dtype/shape, never by value.

    The equation closures carry the declared values as traced leaves, so a new
    value keeps the structure; a rename, a param's shape, a declared function's
    coefficient shape, and a ``pointwise=True`` declaration (the closure calls
    its per-node application) are structural.  An eqset with no declarations
    folds to ``""``.
    """
    params_nv, functions_nv = declared_named_values(eqset)
    parts: list[str] = []
    for kind, nv in (("param", params_nv), ("function", functions_nv)):
        if nv is None:
            continue
        pointwise = frozenset(getattr(nv, "pointwise_names", ()) or ())
        for name, value in sorted(nv.as_dict().items(), key=lambda kv: kv[0]):
            mark = ":pointwise" if name in pointwise else ""
            parts.append(f"{kind}:{name}:{_declared_leaf_fold(value)}{mark}")
    return "|".join(parts)


def content_hash(eqset: EquationSet) -> str:
    """SHA-256 hex digest of the eqset's static spine.

    Optional parts (a component's ``geometry_scale`` binding, non-identity
    Scales, a non-empty residual_scale, families, declared params/functions)
    are folded when present.
    """
    h = hashlib.sha256()
    owners = _build_var_id_to_ref(eqset)

    def put(line: str) -> None:
        h.update(line.encode())
        h.update(b"\n")

    for c in eqset.components:
        scale = getattr(c, "geometry_scale", None)
        put(f"Component:{c.name}:{_domain_id(c.domain)}:scale={scale}")
        if c.operator_factory is not None:
            put("operator_factory=" + _stable_value(c.operator_factory))
        for v in c.variables:
            put(f"Var:{v.name}:{v.shape}:{v.mass}")
        for eq in c.equations:
            put(_eq_id(eq, owners))
        for bc in c.bcs:
            put(_bc_id(bc, owners))
    for g in eqset.globals:
        put(f"Global:{g.name}:{tuple(g.init.shape)}:mass={getattr(g, 'mass', 'algebraic')}")
    for cs in eqset.constraints:
        put(_constraint_id(cs, owners))
    for obs in eqset.observations:
        put(_observation_id(obs, owners))
    for ctl in eqset.controls:
        put(_control_id(ctl, owners))
    for cp in getattr(eqset, "couplings", ()):
        put("Coupling:" + _coupling_id(cp, owners))
    scales = getattr(eqset, "scales", None)
    if scales is not None and scales != identity_scales():
        put(_scales_id(scales))
    residual_scale = getattr(eqset, "residual_scale", None)
    if residual_scale:
        put(_residual_scale_id(residual_scale))
    for fam in getattr(eqset, "families", ()):
        put(_family_id(fam, owners))
    declared = _declared_param_function_id(eqset)
    if declared:
        put(f"Declared:{declared}")
    return h.hexdigest()
