Mathematical Model

Index Sets:
- Let S be the set of shelves, indexed by s, with ShelfID from file_0_view_0.
- Let P be the set of products, indexed by p, with ProductName from file_1_view_0.

Parameters:
- $v_p$: Value of product $p$ (file_1_view_0, column Value)
- $w_p$: Weight of product $p$ (file_1_view_0, column Weight)
- $C_s$: Capacity of shelf $s$ (file_0_view_0, column Capacity)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

Index Sets:
- S (shelves): file_0_view_0, column ShelfID
- P (products): file_1_view_0, column ProductName

Parameters:
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID

Decision Variables:
- $x_{s,p}$: Number of units of product $p$ on shelf $s$ (indexed by ShelfID and ProductName)

All constraints and the objective use these mappings directly from the source data.