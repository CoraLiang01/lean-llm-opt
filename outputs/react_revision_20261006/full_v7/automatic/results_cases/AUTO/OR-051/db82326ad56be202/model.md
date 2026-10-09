#### Mathematical Model

Let:
- $I$ = set of cabinets, indexed by $i$ (CabinetID from file_0_view_0)
- $J$ = set of coffee products, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of cabinet $i$ (Capacity from file_0_view_0)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place in cabinet $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: CabinetID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key CabinetID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: allocation variable for cabinet $i$ (CabinetID) and product $j$ (ProductName)