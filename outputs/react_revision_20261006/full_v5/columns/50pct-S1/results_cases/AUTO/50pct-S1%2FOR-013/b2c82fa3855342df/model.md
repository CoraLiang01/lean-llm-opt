##### Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from all StorageID in file_0_view_0)
- $J$ = set of air conditioner types, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of storage area $i$ (Capacity from file_0_view_0, indexed by StorageID)
- $v_j$ = value of air conditioner type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$ = size (weight) of air conditioner type $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

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

##### Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $c_i$: file_0_view_0, column Capacity, indexed by StorageID
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName
- $x_{ij}$: allocation variable for $(i,j)$

All index sets, parameters, and constraints are defined directly from the supplied data.