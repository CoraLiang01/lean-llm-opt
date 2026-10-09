Mathematical Model

Index Sets:
- Let S be the set of shelves, with elements indexed by s and ShelfID from file_0_view_0.
- Let P be the set of products, with elements indexed by p and ProductName from file_1_view_0.

Parameters:
- $c_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity, key ShelfID)
- $v_p$: Value of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: Weight of product $p$ (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{s,p}$: Number of units of product $p$ to place on shelf $s$; $x_{s,p} \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
$$

Subject to:

Shelf Capacity Constraints:
$$
\sum_{p \in P} w_p \, x_{s,p} \leq c_s \quad \forall s \in S
$$

Integrality and Nonnegativity:
$$
x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

Data Mapping

Index Sets:
- S (shelf): file_0_view_0, column ShelfID
- P (product): file_1_view_0, column ProductName

Parameters:
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

Decision Variables:
- $x_{s,p}$: allocation of product $p$ to shelf $s$ (indexed by ShelfID and ProductName)

All data is mapped directly from the returned CSV files as specified above.