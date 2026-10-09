Mathematical Model

Index Sets:
- $S$: set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

Parameters:
- $c_s$: capacity of shelf $s$ (Capacity from file_0_view_0, keyed by ShelfID)
- $v_p$: value per unit of product $p$ (Value from file_1_view_0, keyed by ProductName)
- $w_p$: weight per unit of product $p$ (Weight from file_1_view_0, keyed by ProductName)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$ (shelves): all ShelfID in file_0_view_0
- $P$ (products): all ProductName in file_1_view_0
- $c_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $x_{sp}$: number of units of product $p$ on shelf $s$ (decision variable, nonnegative integer)