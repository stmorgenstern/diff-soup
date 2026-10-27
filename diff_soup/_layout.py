"""Named device axes and sparse-factor storage policy for a solved problem."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec

from diff_soup._errors import ConfigurationError


class Layout(eqx.Module):
    """Placement of experiments and factorizations on the devices of a mesh.

    As `layout=` on `ds.lower` it splits experiments over the `data` axis, and on `spec.build` it factors a problem over the `model` axis. Settings on a `ds.linear.CuDSS` object, such as `reorder="metis"`, win over the layout's, and the object's `None` leaves them to the layout.

    Args:
        mesh: Mesh from `global_mesh` with axes named `"data"` or `"model"`, or `None` for one device.
        dof_sharding: Stored on the layout and not read by the library.
        reorder: Fill-reducing ordering of cuDSS, where `"metis"` selects METIS and the default `"amd"` keeps cuDSS's own.
        hybrid_memory: If `True`, cuDSS keeps part of the factor in host memory, and a `model` axis across processes takes this from the layout alone.
        hybrid_device_memory_limit_mb: Positive device memory cap in MB for the hybrid mode, which needs `hybrid_memory=True`.
        balance_steps: If `True`, move experiments between devices by their step counts when a device holds several.
        processes: Number of processes the run must have, or `None` for any, and `None` or 1 when the `data` axis spans several devices.
    """

    mesh: Mesh | None = None
    dof_sharding: PartitionSpec | None = None
    reorder: Literal["amd", "metis"] = "amd"
    hybrid_memory: bool = False
    hybrid_device_memory_limit_mb: int | None = None
    balance_steps: bool = False
    processes: int | None = None

    def __check_init__(self) -> None:
        """Reject invalid axes and storage settings before compiling a solve."""
        if self.mesh is not None:
            if not isinstance(self.mesh, Mesh):
                raise ConfigurationError("Layout.mesh must be a jax.sharding.Mesh or None.")
            if not set(self.mesh.axis_names) <= {"data", "model"}:
                raise ConfigurationError("Layout mesh axes must be named 'data' or 'model'.")
            if any(kind != AxisType.Auto for kind in self.mesh.axis_types):
                raise ConfigurationError(
                    "Layout mesh axes must use AxisType.Auto; use ds.global_mesh()."
                )
        if self.reorder not in ("amd", "metis"):
            raise ConfigurationError(
                f"Layout.reorder must be 'amd' or 'metis'; got {self.reorder!r}."
            )
        if self.hybrid_device_memory_limit_mb is not None:
            if not self.hybrid_memory:
                raise ConfigurationError(
                    "Layout.hybrid_device_memory_limit_mb requires hybrid_memory=True."
                )
            if (
                not isinstance(self.hybrid_device_memory_limit_mb, int)
                or self.hybrid_device_memory_limit_mb <= 0
            ):
                raise ConfigurationError(
                    "Layout.hybrid_device_memory_limit_mb must be a positive integer (MB)."
                )
        if self.processes is not None and (
            isinstance(self.processes, bool)
            or not isinstance(self.processes, (int, np.integer))
            or self.processes < 1
        ):
            raise ConfigurationError(
                f"Layout.processes must be a positive integer or None; got {self.processes!r}."
            )
        data = int(self.mesh.shape.get("data", 1)) if self.mesh is not None else 1
        if data > 1 and self.processes is not None and self.processes > 1:
            raise ConfigurationError(
                f"Layout places the lanes over the {data} devices of the mesh's data axis and "
                f"over {self.processes} processes; a layout takes one data placement at a time: "
                "give a mesh whose data axis has one device, or processes=None or 1."
            )

    @property
    def experiment_axis(self) -> str | None:
        """The name of the data axis, when the mesh has one."""
        return "data" if self.mesh is not None and "data" in self.mesh.axis_names else None

    @property
    def dof_size(self) -> int:
        """Number of devices sharing each factorization."""
        return int(self.mesh.shape.get("model", 1)) if self.mesh is not None else 1


def linear_mode(layout: Layout | None) -> str:
    """Select local, single-process MG, or process-distributed MGMN execution."""
    if layout is None or layout.dof_size == 1:
        return "local"
    mesh = layout.mesh
    groups = np.moveaxis(mesh.devices, mesh.axis_names.index("model"), -1).reshape(
        -1, layout.dof_size
    )
    modes = {
        "mgmn" if len({device.process_index for device in group}) > 1 else "mg" for group in groups
    }
    if len(modes) != 1:
        raise ConfigurationError("Every model group must use the same process ownership pattern.")
    return modes.pop()
