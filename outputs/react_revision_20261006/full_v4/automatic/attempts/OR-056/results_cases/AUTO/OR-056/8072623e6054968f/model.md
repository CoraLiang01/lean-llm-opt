##### Mathematical Model

Let:
- $I$ = set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$ = set of vessel types, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of display area $i$ (Capacity from file_0_view_0)
- $v_j$ = value of vessel type $j$ (Value from file_1_view_0)
- $w_j$ = size (weight) of vessel type $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of vessels of type $j$ to be placed in display area $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

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

##### Data Mapping

- $I$: DisplayID from file_0_view_0
- $J$: ProductName from file_1_view_0
- $c_i$: file_0_view_0, column Capacity, key DisplayID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of vessels of type $j$ to be placed in display area $i$ (decision variable, indexed by DisplayID and ProductName)