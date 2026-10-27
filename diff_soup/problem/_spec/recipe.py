"""Reusable Recipe authoring functions with input caching and stable handles."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections import OrderedDict

import numpy as np

from diff_soup.problem._registry import register_parameter_handles
from diff_soup.problem.parameters import parameter_decoder

from .problem_spec import ProblemSpec

#: The input sets a recipe keeps built, the most recently used first to stay.
_BUILD_MEMO_CAPACITY = 128


def _input_key(value):
    """Encode input values without relying on array equality or object identity."""
    if isinstance(value, dict):
        return tuple((key, _input_key(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_input_key(item) for item in value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return type(value).__name__, value
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(
            f"recipe inputs must be scalars, arrays, or containers of them; got {type(value).__name__}"
        )
    return array.dtype.str, array.shape, array.tobytes()


@dataclasses.dataclass(eq=False)
class _TemplateHandle:
    """A named declaration whose definition is supplied by a template build."""

    name: str
    namespace: str = dataclasses.field(metadata={"fingerprint": False})
    declaration: object = None

    @property
    def identity(self):
        """The template name and declared name, independent of its inputs."""
        return self.namespace, self.name

    def __getattr__(self, name):
        declaration = self.__dict__.get("declaration")
        if declaration is None:
            raise ValueError(
                f"recipe handle {self.name!r} has no declaration yet; call template.build with its required inputs before selecting parameters"
            )
        return getattr(declaration, name)

    def __hash__(self):
        return hash(self.identity)

    def __eq__(self, other):
        return self.identity == getattr(other, "identity", None)


class RecipeTemplate:
    """An authoring function returning a ProblemSpec, cached by its inputs.

    Create a template with ``@ds.recipe``. ``build(**inputs)`` returns its
    compiled problem, while ``handle(name)`` names a declaration across builds.
    Required authoring inputs must be supplied before a handle's size or value
    can be inspected. Templates with default arguments can resolve handles
    immediately by building those defaults. Each build keeps its own values:
    a later build with other inputs changes nothing an earlier build returned.
    """

    def __init__(self, author):
        self.author = author
        self.signature = inspect.signature(author)
        self.argument_names = tuple(self.signature.parameters)
        self._builds = OrderedDict()
        self._specs = OrderedDict()
        self._handles = {}
        functools.update_wrapper(self, author)

    def build(self, **inputs):
        """Build once for each distinct set of authoring argument values.

        The same inputs return the same lowered problem; other inputs return
        another problem sharing the compiled structure. A problem keeps the
        values of its own build whatever is built later. The memo holds the
        128 most recently used input sets.
        """
        try:
            bound = self.signature.bind(**inputs)
        except TypeError as error:
            raise ValueError(
                f"recipe {self.__name__!r} inputs do not satisfy its arguments {self.argument_names}: {error}"
            ) from None
        bound.apply_defaults()
        key = _input_key(dict(bound.arguments))
        if key in self._builds:
            self._builds.move_to_end(key)
            self._specs.move_to_end(key)
            return self._builds[key]
        spec = self.author(*bound.args, **bound.kwargs)
        if not isinstance(spec, ProblemSpec):
            raise TypeError(
                f"recipe {self.__name__!r} returned {type(spec).__name__}; the authoring function must return a ProblemSpec"
            )
        # A value given under a scale's name must set the scale; as an authoring
        # argument it would reach the author and leave the scale at its default.
        shadowed = sorted(set(spec._geometry.values()) & set(self.argument_names))
        if shadowed:
            name = shadowed[0]
            raise ValueError(
                f"recipe {self.__name__!r}: the template argument {name!r} has the name of "
                f"the grid scale declared with spec.geometry({name!r}), so a value given as "
                f"{name}=... would go to the recipe and leave the scale at its declared "
                f"value. Rename the argument, and give each experiment's size with "
                f"Experiment(..., inputs={{{name!r}: value}})."
            )
        declarations = {**spec._params, **spec._functions}
        for name, declaration in declarations.items():
            handle = self._handles.setdefault(name, _TemplateHandle(name, self.__name__))
            if handle.declaration is not None:
                size = current_size = 0
                if declaration.learnable or declaration.kind in ("scalar", "array"):
                    size, _ = parameter_decoder(handle)
                    current_size, _ = parameter_decoder(declaration)
                if size != current_size or handle.learnable != declaration.learnable:
                    raise ValueError(
                        f"recipe {self.__name__!r} declaration {name!r} must retain its decoded size and learnable flag across builds"
                    )
            else:
                handle.declaration = declaration
        problem = spec.build()
        self._specs[key] = spec
        self._builds[key] = problem
        while len(self._builds) > _BUILD_MEMO_CAPACITY:
            self._builds.popitem(last=False)
            self._specs.popitem(last=False)
        register_parameter_handles(
            problem, self.__name__, tuple(self._handles[name] for name in declarations)
        )
        return problem

    def handle(self, name):
        """Return the stable handle for a declared parameter or function.

        With no cached build, default arguments are built first. If authoring
        arguments are required, the handle is resolved by the first build.
        A name absent from every cached build raises ValueError.
        """
        if not self._builds:
            required = [
                p
                for p in self.signature.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            if not required:
                self.build()
        declared = {
            name for spec in self._specs.values() for name in (*spec._params, *spec._functions)
        }
        if self._builds and name not in declared:
            raise ValueError(
                f"recipe {self.__name__!r} does not declare {name!r}; declared names: {sorted(declared)}"
            )
        return self._handles.setdefault(name, _TemplateHandle(name, self.__name__))


def recipe(author):
    """Decorate a function returning a ProblemSpec as a cached Recipe template.

    The function's named arguments describe per-build geometry or protocol.
    ``template.build(**inputs)`` compiles it, once per distinct input set, and
    each build holds its own declared values; ``template.handle(name)`` returns
    a handle accepted by every compatible build and by a Study.
    """
    return RecipeTemplate(author)
