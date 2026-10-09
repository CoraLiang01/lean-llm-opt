ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of storage areas (indexed by $i$), from file_0_view_0.StorageID
- $J$: set of air conditioner types (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: capacity of storage area $i$, from file_0_view_0.Capacity
- $v_j$: value per unit of air conditioner type $j$, from file_1_view_0.Value
- $w_j$: size (weight) per unit of air conditioner type $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

DATA MAPPING

- $I$: file_0_view_0.StorageID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by StorageID
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $x_{ij}$: number of units of product $j$ in storage area $i$ (decision variable, integer, nonnegative)