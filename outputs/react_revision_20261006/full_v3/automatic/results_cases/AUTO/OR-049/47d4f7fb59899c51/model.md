Mathematical Model

Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0.ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0.ProductName)

Parameters:
- $c_s$: capacity of shelf $s$ (from file_0_view_0.Capacity)
- $v_p$: value of product $p$ (from file_1_view_0.Value)
- $w_p$: weight of product $p$ (from file_1_view_0.Weight)

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: file_0_view_0.ShelfID
- $P$: file_1_view_0.ProductName
- $c_s$: file_0_view_0.Capacity, keyed by ShelfID
- $v_p$: file_1_view_0.Value, keyed by ProductName
- $w_p$: file_1_view_0.Weight, keyed by ProductName
- $x_{sp}$: decision variable for allocation of product $p$ to shelf $s$