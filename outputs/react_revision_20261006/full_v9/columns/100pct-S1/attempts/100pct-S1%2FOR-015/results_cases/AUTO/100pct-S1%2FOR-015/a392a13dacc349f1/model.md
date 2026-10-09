#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from file_1_view_0, column item_name)
- $J$ = set of shelves, indexed by $j$ (from file_0_view_0, column resource_id)

Parameters:
- $v_i$ = value of one unit of product $i$ (file_1_view_0, item_value)
- $a_i$ = resource requirement (weight) of one unit of product $i$ (file_1_view_0, resource_requirement)
- $c_j$ = capacity of shelf $j$ (file_0_view_0, resource_capacity)

Decision variables:
- $x_{ij}$ = number of units of product $i$ to place on shelf $j$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

Subject to:
\[
\sum_{i \in I} a_i \cdot x_{ij} \leq c_j \qquad \forall j \in J
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$ (products): file_1_view_0, column item_name
- $J$ (shelves): file_0_view_0, column resource_id
- $v_i$: file_1_view_0, column item_value, keyed by item_name
- $a_i$: file_1_view_0, column resource_requirement, keyed by item_name
- $c_j$: file_0_view_0, column resource_capacity, keyed by resource_id
- $x_{ij}$: allocation variable for product $i$ on shelf $j$ (indexed by item_name and resource_id)

All variables $x_{ij}$ are nonnegative integers. Each shelf's total allocated product weight cannot exceed its resource_capacity. The objective is to maximize total value across all shelves and products.