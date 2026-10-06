#### Abstract Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from file_0_view_0, column resource_id)
- $J$ = set of products, indexed by $j$ (from file_1_view_0, column item_name)

Parameters:
- $c_i$ = capacity of shelf $i$ (file_0_view_0, column resource_capacity, key resource_id)
- $v_j$ = value of product $j$ (file_1_view_0, column item_value, key item_name)
- $w_j$ = weight (resource requirement) of product $j$ (file_1_view_0, column resource_requirement, key item_name)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
- Shelf capacity constraints:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Integer and nonnegativity constraints:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$ (shelves): file_0_view_0, column resource_id
- $c_i$: file_0_view_0, column resource_capacity, key resource_id
- $J$ (products): file_1_view_0, column item_name
- $v_j$: file_1_view_0, column item_value, key item_name
- $w_j$: file_1_view_0, column resource_requirement, key item_name

All data is used as returned, preserving original file and row order.