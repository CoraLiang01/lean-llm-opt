Mathematical Optimization Model

Index Sets:
- $I$: set of display areas (DisplayID from file_0_view_0)
- $J$: set of vessel types (ProductName from file_1_view_0)

Parameters:
- $c_i$: capacity of display area $i$ (Capacity from file_0_view_0, indexed by DisplayID)
- $v_j$: value of vessel type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: size (Weight) of vessel type $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{ij}$: number of vessels of type $j$ to be placed in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping:
- $I$: DisplayID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, indexed by DisplayID
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: number of vessels of type $j$ to be placed in display area $i$ (decision variable)