#### Abstract Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from file_0_view_0, column resource_id)
- $J$ = set of products, indexed by $j$ (from file_1_view_0, column item_name)

Parameters:
- $c_i$ = capacity of shelf $i$ (from file_0_view_0, column resource_capacity)
- $v_j$ = value of product $j$ (from file_1_view_0, column item_value)
- $w_j$ = weight (space requirement) of product $j$ (from file_1_view_0, column resource_requirement)

Decision Variables:
- $x_{ij}$ = number of units of product $j$ to place on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

Subject to:
- Shelf capacity constraints:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \quad \forall i \in I
$$

- Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$ (Shelves): file_0_view_0, column resource_id
- $c_i$: file_0_view_0, column resource_capacity, keyed by resource_id
- $J$ (Products): file_1_view_0, column item_name
- $v_j$: file_1_view_0, column item_value, keyed by item_name
- $w_j$: file_1_view_0, column resource_requirement, keyed by item_name

All other columns are ignored. No additional constraints are imposed.