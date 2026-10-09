ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of storage areas (indexed by $i$), with business identifier StorageID from file_0_view_0.
- $J$: set of air conditioner types (indexed by $j$), with business identifier ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of storage area $i$ (Capacity column in file_0_view_0, indexed by StorageID).
- $v_j$: value per unit of air conditioner type $j$ (Value column in file_1_view_0, indexed by ProductName).
- $w_j$: size per unit of air conditioner type $j$ (Weight column in file_1_view_0, indexed by ProductName).

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
- $I$ (storage areas): file_0_view_0, StorageID
- $J$ (air conditioner types): file_1_view_0, ProductName
- $c_i$: file_0_view_0, Capacity (indexed by StorageID)
- $v_j$: file_1_view_0, Value (indexed by ProductName)
- $w_j$: file_1_view_0, Weight (indexed by ProductName)
- $x_{ij}$: number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, $\geq 0$)