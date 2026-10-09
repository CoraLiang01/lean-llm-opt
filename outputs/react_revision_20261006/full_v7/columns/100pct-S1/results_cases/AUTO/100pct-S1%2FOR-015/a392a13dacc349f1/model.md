#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (with business key: item_name from file_1_view_0)
- $J$ = set of shelves, indexed by $j$ (with business key: resource_id from file_0_view_0)

Parameters:
- $v_i$ = value of one unit of product $i$ (item_value from file_1_view_0)
- $w_i$ = weight (resource requirement) of one unit of product $i$ (resource_requirement from file_1_view_0)
- $C_j$ = capacity (weight limit) of shelf $j$ (resource_capacity from file_0_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $i$ to place on shelf $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

Subject to:
\[
\sum_{i \in I} w_i \cdot x_{ij} \leq C_j \qquad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (products): All item_name in file_1_view_0
- $J$ (shelves): All resource_id in file_0_view_0
- $v_i$: file_1_view_0.item_value, keyed by item_name
- $w_i$: file_1_view_0.resource_requirement, keyed by item_name
- $C_j$: file_0_view_0.resource_capacity, keyed by resource_id
- $x_{ij}$: allocation variable for product $i$ (item_name) on shelf $j$ (resource_id), integer and nonnegative

All parameters and index sets are defined by the full set of rows in the respective files as returned above.