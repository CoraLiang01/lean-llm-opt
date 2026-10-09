ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves (indexed by $s$), from file_0_view_0.ShelfID
- $P$: set of products (indexed by $p$), from file_1_view_0.ProductName

Parameters:
- $C_s$: capacity (weight limit) of shelf $s$, from file_0_view_0.Capacity
- $v_p$: value per unit of product $p$, from file_1_view_0.Value
- $w_p$: weight per unit of product $p$, from file_1_view_0.Weight

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

DATA MAPPING

Index Sets:
- $S$: file_0_view_0.ShelfID
- $P$: file_1_view_0.ProductName

Parameters:
- $C_s$: file_0_view_0.Capacity (for shelf $s$)
- $v_p$: file_1_view_0.Value (for product $p$)
- $w_p$: file_1_view_0.Weight (for product $p$)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to place on shelf $s$ (integer, nonnegative)