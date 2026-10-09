Mathematical Model

Index Sets:
- Let $I$ be the set of storage areas, indexed by $i$, with StorageID from file_0_view_0.
- Let $J$ be the set of air conditioner types, indexed by $j$, with ProductName from file_1_view_0.

Parameters:
- $c_i$: Capacity of storage area $i$ (file_0_view_0, column Capacity, key StorageID).
- $v_j$: Value per unit of air conditioner type $j$ (file_1_view_0, column Value, key ProductName).
- $w_j$: Size (Weight) per unit of air conditioner type $j$ (file_1_view_0, column Weight, key ProductName).

Decision Variables:
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:

1. Storage Area Capacity Constraints:
\[
\sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
\]

2. Nonnegativity and Integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Data Mapping

- $I$: file_0_view_0, column StorageID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key StorageID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: Decision variable for allocation of ProductName $j$ to StorageID $i$