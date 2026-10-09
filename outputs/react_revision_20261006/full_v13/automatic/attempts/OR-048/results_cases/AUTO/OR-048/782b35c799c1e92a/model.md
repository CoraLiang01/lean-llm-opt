ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of storage areas (indexed by StorageID from file_0_view_0)
- $J$: set of air conditioner types (indexed by ProductName from file_1_view_0)

Parameters:
- $c_i$: capacity of storage area $i$ (Capacity column, file_0_view_0)
- $v_j$: value of air conditioner type $j$ (Value column, file_1_view_0)
- $w_j$: size (weight) of air conditioner type $j$ (Weight column, file_1_view_0)

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: StorageID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column, file_0_view_0, indexed by StorageID
- $v_j$: Value column, file_1_view_0, indexed by ProductName
- $w_j$: Weight column, file_1_view_0, indexed by ProductName
- $x_{ij}$: number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, nonnegative)