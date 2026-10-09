Mathematical Model

Index Sets:
- Let $I$ be the set of display areas, with elements indexed by $i$ and DisplayID from file_0_view_0.
- Let $J$ be the set of boat types, with elements indexed by $j$ and ProductName from file_1_view_0.

Parameters:
- $c_i$: Capacity of display area $i$ (file_0_view_0, column Capacity, key DisplayID)
- $v_j$: Value of boat type $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$: Weight (size) of boat type $j$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij}$: Number of vessels of type $j$ to be placed in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $I$: file_0_view_0, column DisplayID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key DisplayID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: Number of vessels of type $j$ to be placed in display area $i$ (decision variable, indexed by DisplayID and ProductName)