ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of storage areas (from file_0_view_0, column StorageID)
- $J$: set of air conditioner types (from file_1_view_0, column ProductName)

Parameters:
- $c_i$: capacity of storage area $i$ (from file_0_view_0, column Capacity, indexed by StorageID)
- $v_j$: value of air conditioner type $j$ (from file_1_view_0, column Value, indexed by ProductName)
- $w_j$: size (weight) of air conditioner type $j$ (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

DATA MAPPING

Index Sets:
- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName

Parameters:
- $c_i$: file_0_view_0, column Capacity, indexed by StorageID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$ (indexed by StorageID and ProductName)

All data is mapped directly from the returned CSV files as specified above.