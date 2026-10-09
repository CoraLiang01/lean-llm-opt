### Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (with StorageID from file_0_view_0)
- $J$ = set of air conditioner types, indexed by $j$ (with ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of storage area $i$ (Capacity from file_0_view_0, indexed by StorageID)
- $v_j$ = value per unit of air conditioner type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$ = size (Weight) per unit of air conditioner type $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

### Data Mapping

- $I$ (storage areas): StorageID from file_0_view_0 (capacity.csv)
- $J$ (air conditioner types): ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, indexed by StorageID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: Number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, $\geq 0$)