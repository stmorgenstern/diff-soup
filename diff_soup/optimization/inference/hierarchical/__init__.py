"""Hierarchical and owner-aware estimation for ``diff_soup.inference``.

[TrustRegion][diff_soup.inference.TrustRegion] fits a lowered study on its
lanes, eliminating the levels of its parameter hierarchy from the experiments
up. ``owner_points`` plans compact per-owner point sets.
"""

from diff_soup.inference.hierarchical.owner_points import (
    OwnerChildPointPlan,
    OwnerPointPlan,
    build_owner_point_plans,
    gather_owner_points,
    scatter_owner_points,
)

__all__ = [
    # Owner point plans
    "OwnerPointPlan",
    "OwnerChildPointPlan",
    "build_owner_point_plans",
    "gather_owner_points",
    "scatter_owner_points",
]
