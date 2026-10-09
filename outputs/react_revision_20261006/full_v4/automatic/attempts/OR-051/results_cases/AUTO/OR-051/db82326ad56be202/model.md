##### Mathematical Model

Let:
- $I$ = set of cabinets, indexed by $i$ (with business identifier CabinetID from file_0_view_0)
- $J$ = set of coffee products, indexed by $j$ (with business identifier ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of cabinet $i$ (Capacity from file_0_view_0)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: CabinetID from file_0_view_0
- $J$: ProductName from file_1_view_0
- $c_i$: Capacity from file_0_view_0, indexed by CabinetID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: allocation variable for (CabinetID $i$, ProductName $j$)