ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of cabinets (indexed by $i$), from file_0_view_0.CabinetID
- $J$: set of coffee products (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: capacity of cabinet $i$, from file_0_view_0.Capacity
- $v_j$: value per unit of product $j$, from file_1_view_0.Value
- $w_j$: weight per unit of product $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in cabinet $i$

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

DATA MAPPING

- $I$: file_0_view_0.CabinetID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity (cabinet $i$)
- $v_j$: file_1_view_0.Value (product $j$)
- $w_j$: file_1_view_0.Weight (product $j$)