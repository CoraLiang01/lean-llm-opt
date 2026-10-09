#### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$ = set of products, indexed by $p$ (from all ProductName in file_1_view_0)

Parameters:
- $c_s$ = capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)
- $v_p$ = value per unit of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$ = weight per unit of product $p$ (file_1_view_0, column Weight, key ProductName)

Decision variables:
- $x_{sp}$ = number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
$$

Subject to:
- Shelf capacity constraints:
$$
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
$$

- Integrality and nonnegativity:
$$
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
$$

---

#### Data Mapping

- $S$ (shelves): All ShelfID from file_0_view_0 (capacity.csv)
- $P$ (products): All ProductName from file_1_view_0 (products.csv)
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $x_{sp}$: Decision variable for each $(s,p)$ pair

All indices, parameters, and constraints are mapped directly to the columns and keys as specified above.