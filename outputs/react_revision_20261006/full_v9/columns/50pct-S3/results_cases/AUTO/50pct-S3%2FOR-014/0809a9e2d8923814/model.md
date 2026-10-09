Mathematical Optimization Model

Index Sets:
- $S$: Set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: Set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $v_p$: Value per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: Weight per unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$: Capacity (maximum total weight) of shelf $s$ (from file_0_view_0, column Capacity)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ allocated to shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping:
- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $x_{sp}$: allocation variable for shelf $s$ and product $p$ (decision variable, not in data)