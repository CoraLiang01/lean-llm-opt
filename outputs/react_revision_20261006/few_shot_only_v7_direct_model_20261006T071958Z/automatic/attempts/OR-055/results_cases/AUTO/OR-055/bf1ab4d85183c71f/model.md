ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of display areas (indexed by $i$), from file_0_view_0.DisplayID
- $J$: set of boat types (indexed by $j$), from file_1_view_0.ProductName

Parameters:
- $c_i$: capacity of display area $i$, from file_0_view_0.Capacity
- $v_j$: value of one unit of boat type $j$, from file_1_view_0.Value
- $w_j$: size (weight) of one unit of boat type $j$, from file_1_view_0.Weight

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of boat type $j$ placed in display area $i$

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

- $I$: file_0_view_0.DisplayID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity (for display area $i$)
- $v_j$: file_1_view_0.Value (for boat type $j$)
- $w_j$: file_1_view_0.Weight (for boat type $j$)
- $x_{ij}$: number of units of boat type $j$ in display area $i$ (decision variable)