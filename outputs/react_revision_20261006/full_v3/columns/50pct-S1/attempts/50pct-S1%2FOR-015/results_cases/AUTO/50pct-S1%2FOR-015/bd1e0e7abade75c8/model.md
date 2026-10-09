##### Mathematical Model

Let:
- $I$ = set of shelves, indexed by $i$ (from all resource_id in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all item_name in file_1_view_0)

Parameters:
- $c_i$ = capacity of shelf $i$ (resource_capacity from file_0_view_0)
- $v_j$ = value of product $j$ (item_value from file_1_view_0)
- $w_j$ = weight of product $j$ (resource_requirement from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on shelf $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$ (shelves): All resource_id in file_0_view_0 (capacity.csv)
- $J$ (products): All item_name in file_1_view_0 (products.csv)
- $c_i$: resource_capacity from file_0_view_0, indexed by resource_id
- $v_j$: item_value from file_1_view_0, indexed by item_name
- $w_j$: resource_requirement from file_1_view_0, indexed by item_name
- $x_{ij}$: Integer variable for each $(i,j)$ pair

All indices, parameters, and variables are defined directly from the returned data tables and columns.