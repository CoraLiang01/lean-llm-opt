Mathematical Optimization Model

Index Sets:
- $I$: Set of storage areas, indexed by $i$ (from file_0_view_0, column StorageID)
- $J$: Set of air conditioner types, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $c_i$: Capacity of storage area $i$ (from file_0_view_0, column Capacity)
- $v_j$: Value of air conditioner type $j$ (from file_1_view_0, column Value)
- $w_j$: Size (weight) of air conditioner type $j$ (from file_1_view_0, column Weight)

Decision Variables:
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping:
- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by StorageID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Number of units of ProductName $j$ in StorageID $i$ (decision variable)