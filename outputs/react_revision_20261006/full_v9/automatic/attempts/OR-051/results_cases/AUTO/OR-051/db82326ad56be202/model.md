Mathematical Model

Index Sets:
- $I$: set of cabinets, with elements $i$ corresponding to CabinetID from file_0_view_0 (capacity.csv)
- $J$: set of coffee products, with elements $j$ corresponding to ProductName from file_1_view_0 (products.csv)

Parameters:
- $c_i$: capacity of cabinet $i$ (Capacity column in file_0_view_0)
- $v_j$: value per unit of product $j$ (Value column in file_1_view_0)
- $w_j$: weight per unit of product $j$ (Weight column in file_1_view_0)

Decision Variables:
- $x_{ij}$: number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $I$: CabinetID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column in file_0_view_0, indexed by CabinetID
- $v_j$: Value column in file_1_view_0, indexed by ProductName
- $w_j$: Weight column in file_1_view_0, indexed by ProductName
- $x_{ij}$: number of units of ProductName $j$ in CabinetID $i$ (decision variable)