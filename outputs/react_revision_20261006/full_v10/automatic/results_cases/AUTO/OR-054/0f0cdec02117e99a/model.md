Mathematical Model

Index Sets:
- $S$: set of shelves, indexed by $s$, with ShelfID from file_0_view_0.
- $P$: set of products, indexed by $p$, with ProductName from file_1_view_0.

Parameters:
- $C_s$: capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)
- $v_p$: value per unit of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: weight per unit of product $p$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

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

Data Mapping

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $C_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $x_{sp}$: decision variable, indexed by ShelfID and ProductName

All indices, parameters, and constraints are mapped directly to the supplied CSV columns and business identifiers.