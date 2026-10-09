#### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all item_name in file_1_view_0)
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value of product $j$ (item_value from file_1_view_0)
- $a_j$ = resource requirement (weight) of product $j$ (resource_requirement from file_1_view_0)
- $C_i$ = capacity of shelf $i$ (resource_capacity from file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

- Shelf capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} a_j \, x_{ij} \leq C_i
\]

- Nonnegativity and integrality (for all $i \in I$, $j \in J$):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $I$: All resource_id from file_0_view_0 (capacity.csv, column: resource_id)
- $C_i$: resource_capacity from file_0_view_0 (capacity.csv, column: resource_capacity, key: resource_id)
- $J$: All item_name from file_1_view_0 (products.csv, column: item_name)
- $v_j$: item_value from file_1_view_0 (products.csv, column: item_value, key: item_name)
- $a_j$: resource_requirement from file_1_view_0 (products.csv, column: resource_requirement, key: item_name)
- $x_{ij}$: integer variable for each $(i, j)$ pair

All indices, parameters, and constraints are mapped directly to the columns and keys as specified above.