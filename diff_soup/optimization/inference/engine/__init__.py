"""Batched evaluation of the experiments of a study.

The unit of work is the lane: one experiment solved at one parameter point,
differentiated only against the coordinates it depends on. Lanes of equal
identity, one derivative width and one frame class form a bucket and run
through one compiled program, batched, whatever their parameter values,
initial states, sample times, schedules of operating segments and data. A lane
returns a result whose size is set by its narrow width, with a status of its
own; reductions sum lane results into the blocks of the parameter tree.

The engine has two data placements, chosen by the [Layout][diff_soup.Layout]
a study is lowered with, one at a time. Over the devices of a mesh: a layout
whose mesh has a ``data`` axis of several devices runs every call of every
program split over those devices inside one process (one program per bucket
and mode, mapped over the devices with ``jax.shard_map``), and between
evaluations the units of a call are dealt to the devices by their last trial
step counts, longest first, without a compile. Over processes: the MPI world
the script was launched in (``Layout(processes=...)`` states its size) splits
the lanes among its processes, each evaluating its own on its one device.

| module | holds |
|---|---|
| [_kernel][diff_soup.inference.engine._kernel] | the lane kernel and its modes |
| [_status][diff_soup.inference.engine._status] | the status of a lane |
| [_reducers][diff_soup.inference.engine._reducers] | sums over the parameter tree |
| [_levels][diff_soup.inference.engine._levels] | tables and sums over a parameter hierarchy of any depth, one pass per level |
| [_identity][diff_soup.inference.engine._identity] | the identity of a lane and of a bucket |
| [_capacity][diff_soup.inference.engine._capacity] | frame classes and derivative widths |
| [_layout][diff_soup.inference.engine._layout] | where every lane reads its coordinates in the fitted vector |
| [_inputs][diff_soup.inference.engine._inputs] | the lanes of a group, as the numbers their kernel takes |
| [_plan][diff_soup.inference.engine._plan] | the buckets of a study, built before any compile |
| [_request][diff_soup.inference.engine._request] | a request: mode, points, members, lanes |
| [_order][diff_soup.inference.engine._order] | canonical order, the calls of a program, the order units run in |
| [_balance][diff_soup.inference.engine._balance] | units dealt to the devices of a call, and experiment rows to the lanes of a mesh, by their step counts |
| [_result][diff_soup.inference.engine._result] | the results of a request, in canonical order |
| [_batched][diff_soup.inference.engine._batched] | compiled programs per bucket and mode, and the evaluation of a request |
| [_prepare][diff_soup.inference.engine._prepare] | compiling the programs of a study, and the report of what was compiled |
| [_transport][diff_soup.inference.engine._transport] | the processes of a run and the messages between them, over MPI |
| [_placement][diff_soup.inference.engine._placement] | which process evaluates which lanes: the process placement |
| [_executor][diff_soup.inference.engine._executor] | the lanes of one study evaluated by several processes: coordinator and workers |
| [_ledger][diff_soup.inference.engine._ledger] | what every lane cost so far, for a later run's placement |
"""

from diff_soup.inference.engine._inputs import (
    LaneGroup,
    concatenate_results,
    lane_group,
    lanes_at,
    tree_of,
)
from diff_soup.inference.engine._kernel import (
    MODES,
    LaneInputs,
    LaneProgram,
    LaneResult,
    LaneWork,
    StreamInputs,
    batched_kernel,
)
from diff_soup.inference.engine._ledger import Ledger
from diff_soup.inference.engine._levels import (
    LevelBlocks,
    LevelGradients,
    LevelTables,
    level_tables,
    reduce_level_gradients,
    reduce_levels,
    reduce_member_gradients,
)
from diff_soup.inference.engine._placement import Placement, modelled_costs, place
from diff_soup.inference.engine._plan import Plan
from diff_soup.inference.engine._prepare import PrepareReport, ProgramRecord
from diff_soup.inference.engine._reducers import (
    LaneFailure,
    OwnerBlocks,
    OwnerGradients,
    TreeTables,
    reduce_blocks,
    reduce_gradient,
    tree_tables,
)
from diff_soup.inference.engine._request import Request
from diff_soup.inference.engine._result import Results
from diff_soup.inference.engine._status import KINDS, LaneStatus, failures

__all__ = [
    "KINDS",
    "MODES",
    "LaneFailure",
    "LaneGroup",
    "LaneInputs",
    "LaneProgram",
    "LaneResult",
    "LaneStatus",
    "LaneWork",
    "Ledger",
    "LevelBlocks",
    "LevelGradients",
    "LevelTables",
    "OwnerBlocks",
    "OwnerGradients",
    "Placement",
    "Plan",
    "PrepareReport",
    "ProgramRecord",
    "Request",
    "Results",
    "StreamInputs",
    "TreeTables",
    "batched_kernel",
    "concatenate_results",
    "failures",
    "lane_group",
    "lanes_at",
    "level_tables",
    "modelled_costs",
    "place",
    "reduce_blocks",
    "reduce_gradient",
    "reduce_level_gradients",
    "reduce_levels",
    "reduce_member_gradients",
    "tree_of",
    "tree_tables",
]
