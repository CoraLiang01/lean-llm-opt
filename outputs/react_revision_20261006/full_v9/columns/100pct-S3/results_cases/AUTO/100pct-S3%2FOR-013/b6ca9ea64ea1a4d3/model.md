### Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from all StorageID in file_0_view_0)
- $J$ = set of air conditioner types, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0, column Capacity, key StorageID)
- $v_j$ = value per unit of air conditioner type $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$ = size (weight) per unit of air conditioner type $j$ (file_1_view_0, column Weight, key ProductName)

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

### Data Mapping

- $I$ (storage areas): All StorageID in file_0_view_0 (capacity.csv)
- $J$ (air conditioner types): All ProductName in file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key StorageID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: integer variable for each $(i, j)$ pair

All parameters and index sets are defined by the full set of returned rows in the respective files. No data is omitted or synthesized.