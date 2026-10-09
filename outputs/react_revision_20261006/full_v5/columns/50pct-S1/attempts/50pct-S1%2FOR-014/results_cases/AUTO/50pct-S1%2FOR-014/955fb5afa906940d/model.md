Mathematical Model

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $C_s$: capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: value of product $p$ (from file_1_view_0, column Value)
- $w_p$: weight of product $p$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{s,p}$: decision variable for each $(s,p) \in S \times P$