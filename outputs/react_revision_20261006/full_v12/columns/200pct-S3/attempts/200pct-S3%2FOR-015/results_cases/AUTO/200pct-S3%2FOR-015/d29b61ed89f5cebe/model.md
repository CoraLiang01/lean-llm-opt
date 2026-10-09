### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from file_0_view_0.resource_id)
- $J$ = set of products, indexed by $j$ (from file_1_view_0.item_name)

Parameters:
- $c_i$ = capacity of shelf $i$ (from file_0_view_0.resource_capacity)
- $v_j$ = value per unit of product $j$ (from file_1_view_0.item_value)
- $a_j$ = weight (space requirement) per unit of product $j$ (from file_1_view_0.resource_requirement)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} a_j x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

### Data Mapping

- $I$: file_0_view_0.resource_id
- $J$: file_1_view_0.item_name
- $c_i$: file_0_view_0.resource_capacity (indexed by resource_id)
- $v_j$: file_1_view_0.item_value (indexed by item_name)
- $a_j$: file_1_view_0.resource_requirement (indexed by item_name)
- $x_{ij}$: number of units of product $j$ on shelf $i$ (decision variable, integer, $\geq 0$)