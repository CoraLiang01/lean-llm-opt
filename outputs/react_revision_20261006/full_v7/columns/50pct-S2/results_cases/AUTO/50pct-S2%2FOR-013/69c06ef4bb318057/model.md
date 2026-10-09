#### Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from all StorageID in file_0_view_0)
- $J$ = set of air conditioner types, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0, Capacity, keyed by StorageID)
- $v_j$ = value of air conditioner type $j$ (file_1_view_0, Value, keyed by ProductName)
- $w_j$ = size (weight) of air conditioner type $j$ (file_1_view_0, Weight, keyed by ProductName)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

#### Data Mapping

- $I$: All StorageID from file_0_view_0 (capacity.csv), column StorageID
- $J$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by StorageID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: integer, for each $(i,j) \in I \times J$