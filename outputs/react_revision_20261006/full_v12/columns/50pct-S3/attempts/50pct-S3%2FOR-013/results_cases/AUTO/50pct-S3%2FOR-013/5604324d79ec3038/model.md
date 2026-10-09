### Mathematical Model

Let:
- $I$ = set of storage areas (indexed by $i$), with StorageID from capacity.csv
- $J$ = set of air conditioner types (indexed by $j$), with ProductName from products.csv

Parameters:
- $c_i$ = capacity of storage area $i$ (Capacity, from capacity.csv)
- $v_j$ = value of air conditioner type $j$ (Value, from products.csv)
- $w_j$ = size (Weight) of air conditioner type $j$ (Weight, from products.csv)

Decision variables:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \quad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

### Data Mapping

- $I$: StorageID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, indexed by StorageID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: Number of units of ProductName $j$ in StorageID $i$ (decision variable, integer, $\geq 0$)