# Domains

A domain is what `spec.domain(name, domain)` registers, and its
discretization decides how the operator bank evaluates each operator. See
[domains and discretizations](../../overview/modeling/domains.md).

!!! tip "Recommended defaults"
    Structured problems use the grid factories with the default
    `method="fvm"`. For finite elements, build a `TaggedMesh` so boundary
    conditions can select boundaries by name. The mesh factories return one
    with `with_boundary_tags=True`.

---

### Grid factories

::: diff_soup.domains.cartesian_grid_1d

::: diff_soup.domains.cartesian_grid_2d

::: diff_soup.domains.cartesian_grid_3d

::: diff_soup.domains.cylindrical_grid_1d

::: diff_soup.domains.spherical_grid_1d

::: diff_soup.domains.staggered_grid_2d

::: diff_soup.domains.staggered_grid_3d

---

### Mesh factories

::: diff_soup.domains.interval_mesh

::: diff_soup.domains.cylindrical_interval_mesh

::: diff_soup.domains.spherical_interval_mesh

::: diff_soup.domains.rectangle_mesh

::: diff_soup.domains.box_mesh

::: diff_soup.domains.geometry.polygons_to_tagged_mesh

---

### Mesh types

::: diff_soup.domains.Mesh
    options:
      members: false

::: diff_soup.domains.TaggedMesh
    options:
      members:
        - facets_for_tag
        - facets_for_name
        - cell_indicator

---

??? abstract "Domain types"

    The details matter only when extending the library.

    ::: diff_soup.domains.Domain
        options:
          members:
            - dim
            - num_nodes
            - node_coordinates
            - discretization_type

    ::: diff_soup.domains.Grid1D
        options:
          members:
            - nodes
            - faces
            - cell_volumes
            - coordinates
            - meshgrid
            - scaled

    ::: diff_soup.domains.GridND
        options:
          members:
            - axes
            - shape
            - coordinates
            - meshgrid
            - node_coordinates

    ::: diff_soup.domains.StaggeredGrid
        options:
          members:
            - cell_grid
            - shape
            - face_shape

    ::: diff_soup.domains.CoordSystem
        options:
          members: false

    ::: diff_soup.domains.DiscreteField
        options:
          members: false
