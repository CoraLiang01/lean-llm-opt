ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of display areas (indexed by $i$), with business identifier DisplayID from file_0_view_0.
- $J$: set of boat types (indexed by $j$), with business identifier ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of display area $i$ (Capacity column, file_0_view_0, key DisplayID).
- $v_j$: value of one unit of boat type $j$ (Value column, file_1_view_0, key ProductName).
- $w_j$: size of one unit of boat type $j$ (Weight column, file_1_view_0, key ProductName).

Decision Variables:
- $x_{ij}$: number of units of boat type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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
- $I$ (display areas): DisplayID from file_0_view_0 (capacity.csv)
- $J$ (boat types): ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, key DisplayID
- $v_j$: Value from file_1_view_0, key ProductName
- $w_j$: Weight from file_1_view_0, key ProductName