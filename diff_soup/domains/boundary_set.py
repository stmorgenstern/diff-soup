# Public boundary-set selector value type for the Recipe API.
"""Boundary-set selectors: name *where* a boundary condition or integral applies.

A [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] is a small, immutable expression tree that *describes* a
selection of a domain's boundary without holding any domain data.  You build one
with the [tags][diff_soup.domains.boundary_set.tags] constructor and combine selections with the ``|`` (union)
and ``-`` (difference) operators::

    import diff_soup as ds

    left        = ds.tags("left")
    left_right  = ds.tags("left", "right")          # facets named left OR right
    walls       = ds.tags("walls") - ds.tags("inlet")  # walls minus the inlet
    everything  = ds.tags("walls") | ds.tags("inlet")  # walls together with inlet

The value records only tag-name strings, raw node indices, and child
[BoundarySet][diff_soup.domains.boundary_set.BoundarySet] nodes.  It is frozen, hashable, and compares by value, so it
is safe to use as a dict key, deduplicate, or embed in another frozen record.

Turning a selection into concrete facets/nodes requires a domain (a mesh), so
that step happens later, during lowering — not here.  To evaluate the tree
without inspecting its internal shape, use [BoundarySet.reduce][diff_soup.domains.boundary_set.BoundarySet.reduce], which
folds the tree with one callback per node kind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, TypeVar

import numpy as np

__all__ = ["BoundarySet", "from_nodes", "tags"]

T = TypeVar("T")


@dataclass(frozen=True)
class BoundarySet:
    """A boundary selector, built from tag names or node indices.

    Build one with [`ds.tags`][diff_soup.problem.tags] or `BoundarySet.from_nodes`. Combine selectors with `|` (union) and `-` (difference), and pass the result as `on=` to a `spec.bc` declaration. It is frozen and hashable.
    """

    op: str
    names: tuple[str, ...] = ()
    nodes: tuple[int, ...] = ()
    left: Optional["BoundarySet"] = None
    right: Optional["BoundarySet"] = None

    def __or__(self, other: "BoundarySet") -> "BoundarySet":
        """Union: select boundary in ``self`` OR ``other`` (``self | other``).

        Args:
            other: Another [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] to union with.

        Returns:
            A new [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] selecting the union of the two operands.

        Raises:
            TypeError: If ``other`` is not a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet].
        """
        if not isinstance(other, BoundarySet):
            raise TypeError(
                "BoundarySet union (|) requires both operands to be BoundarySet; "
                f"got {type(other).__name__} on the right. Build the operand with "
                "ds.tags(...) (or from_nodes(...)) first."
            )
        return BoundarySet(op="union", left=self, right=other)

    def __sub__(self, other: "BoundarySet") -> "BoundarySet":
        """Difference: select boundary in ``self`` but NOT ``other`` (``self - other``).

        Args:
            other: Another [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] to subtract.

        Returns:
            A new [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] selecting ``self`` with ``other`` removed.

        Raises:
            TypeError: If ``other`` is not a [BoundarySet][diff_soup.domains.boundary_set.BoundarySet].
        """
        if not isinstance(other, BoundarySet):
            raise TypeError(
                "BoundarySet difference (-) requires both operands to be BoundarySet; "
                f"got {type(other).__name__} on the right. Build the operand with "
                "ds.tags(...) (or from_nodes(...)) first."
            )
        return BoundarySet(op="difference", left=self, right=other)

    def reduce(
        self,
        *,
        on_tags: Callable[[tuple[str, ...]], T],
        on_nodes: Callable[[tuple[int, ...]], T],
        on_union: Callable[[T, T], T],
        on_difference: Callable[[T, T], T],
    ) -> T:
        """Fold the selector expression into one value.

        Each tag leaf maps through `on_tags`, each node leaf through `on_nodes`, and each union or difference combines the values of its operands. Use it to resolve a selector against a domain, for example into a set of node indices.

        Args:
            on_tags: Maps a tuple of tag names to a value.
            on_nodes: Maps a tuple of node indices to a value.
            on_union: Combines the values of the two operands of a union.
            on_difference: Combines the values of the left and right operands of a difference.

        Returns:
            The value of the whole expression.
        """
        if self.op == "tags":
            return on_tags(self.names)
        if self.op == "nodes":
            return on_nodes(self.nodes)
        left_value = self.left.reduce(
            on_tags=on_tags,
            on_nodes=on_nodes,
            on_union=on_union,
            on_difference=on_difference,
        )
        right_value = self.right.reduce(
            on_tags=on_tags,
            on_nodes=on_nodes,
            on_union=on_union,
            on_difference=on_difference,
        )
        if self.op == "union":
            return on_union(left_value, right_value)
        return on_difference(left_value, right_value)

    @classmethod
    def from_nodes(cls, indices) -> "BoundarySet":
        """Build a selector from node indices or a boolean mask.

        Args:
            indices: Node indices, or a boolean mask that selects its `True` positions, flattened in either case.

        Returns:
            A selector of those nodes.
        """
        return from_nodes(indices)


def tags(*names: str) -> BoundarySet:
    """Select boundary locations by tag name.

    Several names select their union. On a mesh the names are those of its [TaggedMesh][diff_soup.domains.TaggedMesh], and a one-dimensional grid names its last node `"surface"`.

    Args:
        *names: One or more non-empty tag names.

    Returns:
        A [BoundarySet][diff_soup.problem.BoundarySet], which combines with others through `|` and `-`.

    !!! example

        ```python
        spec.bc.dirichlet("u", on=ds.tags("boundary") - ds.tags("rim"), value=1.0)
        ```
    """
    if len(names) == 0:
        raise ValueError("tags(...) requires at least one tag name; pass one or more strings.")
    for name in names:
        if not isinstance(name, str) or name == "":
            raise ValueError(
                "tags(...) names must each be a non-empty string; "
                f"got {name!r}. Pass the facet-tag name(s) as strings."
            )
    return BoundarySet(op="tags", names=tuple(names))


def from_nodes(indices) -> BoundarySet:
    """Select boundary nodes by raw index (or by a boolean mask).

    Use this when you want to pin specific nodes directly rather than name a
    tagged region.  ``indices`` may be an array-like of integer node indices, or
    a boolean mask over the domain's nodes — in which case its ``True`` positions
    are selected (via `numpy.nonzero`, matching the boundary-spec IR).  The
    indices are normalized to a tuple of Python ints so the result stays frozen
    and hashable.

    Args:
        indices: An array-like of integer node indices, or a boolean mask whose
            ``True`` entries name the selected nodes.

    Returns:
        A [BoundarySet][diff_soup.domains.boundary_set.BoundarySet] ``nodes`` leaf holding the selected node indices.
    """
    arr0 = np.asarray(indices)
    if arr0.dtype == bool:
        # A boolean mask selects its True positions; casting it straight to int
        # would coerce True/False to the literal indices {0, 1} and silently pin
        # the wrong nodes.
        indices = np.nonzero(arr0.reshape(-1))[0]
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    return BoundarySet(op="nodes", nodes=tuple(int(x) for x in arr))
