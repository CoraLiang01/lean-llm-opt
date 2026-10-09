### Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from file_0_view_0, column StorageID)
- $J$ = set of air conditioner types, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0, column Capacity)
- $v_j$ = value per unit of air conditioner type $j$ (file_1_view_0, column Value)
- $w_j$ = size (weight) per unit of air conditioner type $j$ (file_1_view_0, column Weight)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

---

### Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by StorageID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, $\geq 0$)