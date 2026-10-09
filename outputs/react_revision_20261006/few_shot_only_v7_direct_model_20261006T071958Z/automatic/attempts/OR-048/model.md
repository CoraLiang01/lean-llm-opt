ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of storage areas (indexed by $i$), from file_0_view_0.StorageID
- $J$: set of air conditioner types (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: capacity of storage area $i$, from file_0_view_0.Capacity
- $v_j$: value of air conditioner type $j$, from file_1_view_0.Value
- $w_j$: size (weight) of air conditioner type $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of air conditioner type $j$ placed in storage area $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

DATA MAPPING

- $I$ (storage areas): file_0_view_0.StorageID
- $J$ (air conditioner types): file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by StorageID $i$
- $v_j$: file_1_view_0.Value, keyed by ProductName $j$
- $w_j$: file_1_view_0.Weight, keyed by ProductName $j$