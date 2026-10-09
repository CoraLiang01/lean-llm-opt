Mathematical Model

Index Sets:
- Let $I$ be the set of storage areas, with each $i \in I$ identified by StorageID from file_0_view_0.
- Let $J$ be the set of air conditioner types, with each $j \in J$ identified by ProductName from file_1_view_0.

Parameters:
- $c_i$: Capacity of storage area $i$ (Capacity column in file_0_view_0).
- $v_j$: Value of air conditioner type $j$ (Value column in file_1_view_0).
- $w_j$: Size (Weight) of air conditioner type $j$ (Weight column in file_1_view_0).

Decision Variables:
- $x_{ij}$: Number of units of air conditioner type $j$ placed in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: StorageID in file_0_view_0 (capacity.csv)
- $J$: ProductName in file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, keyed by StorageID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Number of units of product $j$ in storage area $i$ (decision variable, integer, nonnegative)