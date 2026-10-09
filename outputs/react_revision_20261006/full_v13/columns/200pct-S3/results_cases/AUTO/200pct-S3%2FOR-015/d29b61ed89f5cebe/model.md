#### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from file_0_view_0.resource_id)
- $J$ = set of products, indexed by $j$ (from file_1_view_0.item_name)

Parameters:
- $c_i$ = capacity of shelf $i$ (from file_0_view_0.resource_capacity)
- $v_j$ = value per unit of product $j$ (from file_1_view_0.item_value)
- $a_j$ = resource requirement (weight/space) per unit of product $j$ (from file_1_view_0.resource_requirement)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

#### Data Mapping

- $I$ (shelves): file_0_view_0.resource_id
- $J$ (products): file_1_view_0.item_name
- $c_i$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_j$: file_1_view_0.item_value, keyed by item_name
- $a_j$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{ij}$: allocation variable for (resource_id $i$, item_name $j$)

All indices, parameters, and constraints are mapped directly from the current CSV data as described above.