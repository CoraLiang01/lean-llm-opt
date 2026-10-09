##### Mathematical Model

Let:
- $I$ = set of storage areas (indexed by $i$), with StorageID from file_0_view_0
- $J$ = set of air conditioner types (indexed by $j$), with ProductName from file_1_view_0

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0, Capacity)
- $v_j$ = value of air conditioner type $j$ (file_1_view_0, Value)
- $w_j$ = size (weight) of air conditioner type $j$ (file_1_view_0, Weight)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

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

---

##### Data Mapping

- $I$ (storage areas): file_0_view_0, StorageID
- $J$ (air conditioner types): file_1_view_0, ProductName
- $c_i$: file_0_view_0, Capacity, keyed by StorageID
- $v_j$: file_1_view_0, Value, keyed by ProductName
- $w_j$: file_1_view_0, Weight, keyed by ProductName
- $x_{ij}$: number of units of air conditioner type $j$ in storage area $i$ (decision variable, integer, $\geq 0$)