ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: value of product $p$ (from file_1_view_0, column Value)
- $w_p$: weight of product $p$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to place on shelf $s$

Objective:
$$
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
$$

Subject to:
- Shelf capacity constraints:
$$
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \quad \forall s \in S
$$

- Integrality and nonnegativity:
$$
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
$$

DATA MAPPING

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $c_s$: file_0_view_0, column Capacity, indexed by ShelfID
- $v_p$: file_1_view_0, column Value, indexed by ProductName
- $w_p$: file_1_view_0, column Weight, indexed by ProductName