ABSTRACT MATHEMATICAL MODEL

Sets:
- $S$: set of storage areas (indexed by $s$), from capacity.csv [StorageID]
- $P$: set of air conditioner types (indexed by $p$), from products.csv [ProductName]

Parameters:
- $c_s$: capacity of storage area $s$, from capacity.csv [Capacity]
- $v_p$: value of air conditioner type $p$, from products.csv [Value]
- $w_p$: size (weight) of air conditioner type $p$, from products.csv [Weight]

Decision Variables:
- $x_{sp}$: number of units of air conditioner type $p$ placed in storage area $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping:

- $S$ (storage areas): file_0_view_0 [StorageID]
- $c_s$: file_0_view_0 [Capacity], keyed by [StorageID]
- $P$ (air conditioner types): file_1_view_0 [ProductName]
- $v_p$: file_1_view_0 [Value], keyed by [ProductName]
- $w_p$: file_1_view_0 [Weight], keyed by [ProductName]