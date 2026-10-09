Mathematical Model

Index Sets:
- $I$: set of shelves (resources), indexed by $i$, with business identifier resource_id from file_0_view_0.
- $J$: set of products, indexed by $j$, with business identifier item_name from file_1_view_0.

Parameters:
- $c_i$: capacity of shelf $i$, parameter resource_capacity from file_0_view_0.
- $v_j$: value per unit of product $j$, parameter item_value from file_1_view_0.
- $a_j$: weight (space requirement) per unit of product $j$, parameter resource_requirement from file_1_view_0.

Decision Variables:
- $x_{ij}$: number of units of product $j$ to place on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$ (shelves): resource_id from file_0_view_0
- $J$ (products): item_name from file_1_view_0
- $c_i$: resource_capacity from file_0_view_0, indexed by resource_id
- $v_j$: item_value from file_1_view_0, indexed by item_name
- $a_j$: resource_requirement from file_1_view_0, indexed by item_name
- $x_{ij}$: number of units of product $j$ on shelf $i$, indexed by (resource_id, item_name)

All parameters and index sets are defined by the current rows and columns of the returned CSV files.