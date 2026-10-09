ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of storage areas (indexed by $i$), with StorageID from file_0_view_0.
- $J$: set of air conditioner types (indexed by $j$), with ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of storage area $i$ (Capacity, file_0_view_0, StorageID $i$).
- $v_j$: value per unit of air conditioner type $j$ (Value, file_1_view_0, ProductName $j$).
- $w_j$: size (weight) per unit of air conditioner type $j$ (Weight, file_1_view_0, ProductName $j$).

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

Data Mapping:
- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key StorageID $i$
- $v_j$: file_1_view_0, column Value, key ProductName $j$
- $w_j$: file_1_view_0, column Weight, key ProductName $j$