Mathematical Model

Index Sets:
- $I$: set of shelves (from file_0_view_0.resource_id)
- $J$: set of products (from file_1_view_0.item_name)

Parameters:
- $C_i$: capacity of shelf $i$ (from file_0_view_0.resource_capacity, indexed by resource_id)
- $v_j$: value of product $j$ (from file_1_view_0.item_value, indexed by item_name)
- $w_j$: weight (resource requirement) of product $j$ (from file_1_view_0.resource_requirement, indexed by item_name)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: file_0_view_0.resource_id
- $J$: file_1_view_0.item_name
- $C_i$: file_0_view_0.resource_capacity, indexed by resource_id
- $v_j$: file_1_view_0.item_value, indexed by item_name
- $w_j$: file_1_view_0.resource_requirement, indexed by item_name
- $x_{ij}$: number of units of product $j$ placed on shelf $i$ (decision variable, integer, nonnegative)