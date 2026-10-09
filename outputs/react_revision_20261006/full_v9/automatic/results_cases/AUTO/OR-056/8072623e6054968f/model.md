Mathematical Model

Sets:
- $I$: set of display areas (indexed by $i$), with business identifier DisplayID from file_0_view_0.
- $J$: set of boat types (indexed by $j$), with business identifier ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of display area $i$ (file_0_view_0, column Capacity, key DisplayID).
- $v_j$: value of boat type $j$ (file_1_view_0, column Value, key ProductName).
- $w_j$: size (weight) of boat type $j$ (file_1_view_0, column Weight, key ProductName).

Decision Variables:
- $x_{ij}$: number of boats of type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

- $I$: DisplayID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: file_0_view_0, column Capacity, key DisplayID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of boats of type $j$ in display area $i$ (decision variable, nonnegative integer)