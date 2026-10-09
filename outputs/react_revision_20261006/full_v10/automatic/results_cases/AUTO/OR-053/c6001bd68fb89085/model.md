Mathematical Model

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$: value per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: weight per unit of product $p$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{s,p}$: number of units of product $p$ to place on shelf $s$ ($x_{s,p} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{s,p} \leq c_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $c_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{s,p}$: allocation variable for shelf $s$ and product $p$ (integer, nonnegative)