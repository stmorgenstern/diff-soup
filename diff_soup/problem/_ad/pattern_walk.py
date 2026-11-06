"""The walk behind a Jacobian pattern: a jaxpr lowered to integers, its known part, its flows.

A jaxpr is lowered once into blocks of operations on integer variables, with every call
primitive inlined, so one walk and one compiled program serve the whole function.
A variable is known when it holds one value whatever the inputs are: a literal, a
closed-over array, or an operation of known variables only. Every known value an
operation on tangents reads is computed by one ``jax.jit`` program, and the walk then
carries a [flow][diff_soup.problem._ad.flows] per variable through the rules.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal

from diff_soup._warnings import warn_user
from diff_soup.problem._ad import flows as F

# Primitives whose jaxpr the lowering splices into its caller.
_INLINED = {
    "jit": "jaxpr",
    "pjit": "jaxpr",
    "closed_call": "call_jaxpr",
    "core_call": "call_jaxpr",
    "checkpoint": "jaxpr",
    "remat": "jaxpr",
    "custom_jvp_call": "call_jaxpr",
    "custom_vjp_call": "call_jaxpr",
}
# Parameters holding the jaxprs a rule walks, per primitive.
_NESTED = {
    "cond": lambda params: params["branches"],
    "scan": lambda params: (params["jaxpr"],),
    "while": lambda params: (params["cond_jaxpr"], params["body_jaxpr"]),
    "custom_linear_solve": lambda params: (params["jaxprs"].matvec,),
}


class Lit(NamedTuple):
    """A literal operand."""

    value: np.ndarray
    shape: tuple


class Op(NamedTuple):
    """One operation of a lowered block.

    Attributes:
        primitive: The JAX primitive.
        params: Its parameters, as in the jaxpr.
        invars: Its operands: variable numbers or literals.
        outvars: Its results' variable numbers.
        blocks: The lowered jaxprs a rule walks (branches, a loop body), else empty.
        eqn: The jaxpr equation, which the known part evaluates.
    """

    primitive: object
    params: dict
    invars: tuple
    outvars: tuple
    blocks: tuple
    eqn: object


class Block(NamedTuple):
    """One lowered jaxpr: its inputs, its operations and its outputs."""

    invars: tuple
    ops: tuple
    outvars: tuple


class Program:
    """A function's jaxpr lowered for the walk, with its known values computed."""

    def __init__(self, closed: ClosedJaxpr):
        """Lower ``closed``, find its known variables and compute those the walk reads."""
        self._count = itertools.count()
        self.shapes: dict[int, tuple] = {}
        self.consts: dict[int, object] = {}
        self.known: set[int] = set()
        self.values: dict[int, np.ndarray] = {}
        self.top = self._lower(closed.jaxpr, closed.consts, None)
        needed: list[int] = []
        self._mark(self.top, set(), needed)
        self._evaluate(sorted(set(needed)))

    def _new(self, shape) -> int:
        number = next(self._count)
        self.shapes[number] = tuple(shape)
        return number

    def _lower(self, jaxpr: Jaxpr, consts, operands) -> Block:
        """Lower ``jaxpr``; ``operands`` replaces its inputs when it is inlined."""
        names: dict = {}

        def atom(var):
            if isinstance(var, Literal):
                value = np.asarray(var.val)
                return Lit(value, value.shape)
            return names[var]

        for var, value in zip(jaxpr.constvars, consts):
            names[var] = self._new(var.aval.shape)
            self.consts[names[var]] = value
        if operands is None:
            invars = tuple(self._new(var.aval.shape) for var in jaxpr.invars)
        else:
            invars = tuple(operands)
        names.update(zip(jaxpr.invars, invars))
        ops = []
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            operands_here = [atom(var) for var in eqn.invars]
            if name in _INLINED:
                block = self._lower(*_opened(eqn.params[_INLINED[name]]), operands_here)
                ops.extend(block.ops)
                names.update(zip(eqn.outvars, block.outvars))
                continue
            blocks = ()
            if name in _NESTED:
                blocks = tuple(
                    self._lower(*_opened(sub), None) for sub in _NESTED[name](eqn.params)
                )
            outvars = tuple(self._new(var.aval.shape) for var in eqn.outvars)
            names.update(zip(eqn.outvars, outvars))
            ops.append(Op(eqn.primitive, eqn.params, tuple(operands_here), outvars, blocks, eqn))
        return Block(invars, tuple(ops), tuple(atom(var) for var in jaxpr.outvars))

    def is_known(self, atom) -> bool:
        """Whether ``atom`` holds one value whatever the function's inputs are."""
        return isinstance(atom, Lit) or atom in self.known or atom in self.consts

    def _mark(self, block: Block, known_inputs: set, needed: list):
        """Mark the known variables of ``block`` and list those an unknown operation reads."""
        self.known.update(known_inputs)
        for op in block.ops:
            if all(self.is_known(atom) for atom in op.invars) and not op.eqn.effects:
                self.known.update(op.outvars)
                continue
            needed.extend(a for a in op.invars if not isinstance(a, Lit) and self.is_known(a))
            for sub in op.blocks:
                inputs = _known_block_inputs(op, sub, self.is_known)
                self._mark(sub, set(inputs), needed)

    def _evaluate(self, needed: list[int]):
        """Compute the ``needed`` known values in one compiled program."""
        if not needed:
            return
        const_ids = sorted(self.consts)

        def run(*const_values):
            env = dict(zip(const_ids, const_values))
            self._run(self.top, env)
            return [env[number] for number in needed]

        results = jax.jit(run)(*(self.consts[number] for number in const_ids))
        self.values.update(zip(needed, (np.asarray(result) for result in results)))

    def _run(self, block: Block, env: dict):
        """Evaluate the known operations of ``block`` into ``env``, under a trace."""

        def read(atom):
            return atom.value if isinstance(atom, Lit) else env[atom]

        for op in block.ops:
            if all(self.is_known(atom) for atom in op.invars) and not op.eqn.effects:
                params = op.primitive.get_bind_params(op.params)
                result = op.primitive.bind(*map(read, op.invars), **params)
                results = result if op.primitive.multiple_results else (result,)
                env.update(zip(op.outvars, results))
                continue
            for sub in op.blocks:
                for inner, outer in _known_block_inputs(op, sub, self.is_known).items():
                    env[inner] = read(outer)
                self._run(sub, env)

    def value(self, atom):
        """The value ``atom`` holds whatever the inputs are, or ``None``."""
        if isinstance(atom, Lit):
            return atom.value
        return self.values.get(atom)


def _opened(jaxpr) -> tuple:
    """``(jaxpr, consts)`` of a closed or an open jaxpr."""
    if isinstance(jaxpr, ClosedJaxpr):
        return jaxpr.jaxpr, jaxpr.consts
    return jaxpr, ()


def scan_counts(params) -> tuple[int, int]:
    """The number of constants and of carries of a ``scan``."""
    if "num_consts" in params:
        return int(params["num_consts"]), int(params["num_carry"])
    consts, carry, _ = params["ft_in"].unpack()
    return len(list(consts.vals)), len(list(carry.vals))


def _known_block_inputs(op: Op, block: Block, is_known) -> dict:
    """The inputs of a nested ``block`` that hold one value, with the operand each one reads."""
    name = op.primitive.name
    if name == "cond":
        pairs = zip(block.invars, op.invars[1:])
    elif name == "scan":
        pairs = zip(block.invars[: scan_counts(op.params)[0]], op.invars)
    elif name == "while":
        cond_n, body_n = op.params["cond_nconsts"], op.params["body_nconsts"]
        if block is op.blocks[0]:
            pairs = zip(block.invars[:cond_n], op.invars[:cond_n])
        else:
            pairs = zip(block.invars[:body_n], op.invars[cond_n : cond_n + body_n])
    elif name == "custom_linear_solve":
        count = op.params["const_lengths"].matvec
        pairs = zip(block.invars[:count], op.invars[:count])
    else:
        pairs = ()
    return {inner: outer for inner, outer in pairs if is_known(outer)}


PatternRule = Callable[["Walk", Op], list]
_RULES: dict[str, PatternRule] = {}


def register_pattern_rule(primitive, rule: PatternRule) -> None:
    """Register how tangents cross ``primitive`` for the pattern walk.

    Args:
        primitive: A JAX primitive or its name.
        rule: ``rule(walk, op)`` returning one flow (or ``None``) per output of ``op``,
            from ``walk.flow`` and ``walk.value`` of its operands.
    """
    _RULES[getattr(primitive, "name", primitive)] = rule


class Walk:
    """The flows of one walk of a [Program][diff_soup.problem._ad.pattern_walk.Program]."""

    def __init__(self, program: Program, seeds: int):
        """Start a walk with ``seeds`` columns and no flow."""
        self.program = program
        self.seeds = seeds
        self.flows: dict[int, object] = {}
        self.fallbacks: set = set()

    def flow(self, atom):
        """The flow reaching ``atom``, or ``None`` when no tangent does."""
        return None if isinstance(atom, Lit) else self.flows.get(atom)

    def value(self, atom):
        """The value ``atom`` holds whatever the inputs are, or ``None``."""
        return self.program.value(atom)

    def shape(self, atom) -> tuple:
        """The shape of ``atom``."""
        return atom.shape if isinstance(atom, Lit) else self.program.shapes[atom]

    def size(self, atom) -> int:
        """The number of elements of ``atom``."""
        return int(np.prod(self.shape(atom), dtype=np.int64))

    def tables(self, atoms):
        """Row numbers of ``atoms`` in one stacked flow; an array no tangent reaches holds -1."""
        tables, blocks, start = [], [], 0
        for atom in atoms:
            flow = self.flow(atom)
            if flow is None:
                tables.append(np.full(self.shape(atom), -1, dtype=np.int64))
            else:
                tables.append(np.arange(start, start + flow.shape[0]).reshape(self.shape(atom)))
                blocks.append(flow)
                start += flow.shape[0]
        return tables, F.stack(blocks, self.seeds)

    def fallback(self, op: Op, reason: str) -> list:
        """Every output of ``op`` on every input, with a warning the first time."""
        key = (op.primitive.name, reason)
        if key not in self.fallbacks:
            self.fallbacks.add(key)
            warn_user(
                f"tangent_pattern: {op.primitive.name} {reason}, so each of its outputs is "
                "declared to read each of its inputs."
            )
        reaching = [self.flow(atom) for atom in op.invars if self.flow(atom) is not None]
        return F.anywhere(reaching, [self.size(var) for var in op.outvars], self.seeds)

    def block(self, block: Block, inputs) -> list:
        """Walk ``block`` from the flows of its ``inputs``; return its outputs' flows."""
        for var, flow in zip(block.invars, inputs):
            if flow is None:
                self.flows.pop(var, None)
            else:
                self.flows[var] = flow
        for op in block.ops:
            if not any(self.flow(atom) is not None for atom in op.invars):
                continue
            rule = _RULES.get(op.primitive.name)
            if rule is None:
                outputs = self.fallback(op, "has no pattern rule")
            else:
                outputs = rule(self, op)
            for var, flow in zip(op.outvars, outputs):
                if flow is None:
                    self.flows.pop(var, None)
                else:
                    self.flows[var] = flow
        return [self.flow(atom) for atom in block.outvars]
