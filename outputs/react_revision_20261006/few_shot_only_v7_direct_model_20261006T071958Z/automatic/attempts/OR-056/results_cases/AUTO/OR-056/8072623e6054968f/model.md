ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of display areas (indexed by $i$), corresponding to all DisplayID in file_0_view_0.
- $J$: set of boat types (indexed by $j$), corresponding to all ProductName in file_1_view_0.

Parameters:
- $c_i$: capacity of display area $i$ (from file_0_view_0, column Capacity, indexed by DisplayID).
- $v_j$: value of boat type $j$ (from file_1_view_0, column Value, indexed by ProductName).
- $w_j$: weight (size) of boat type $j$ (from file_1_view_0, column Weight, indexed by ProductName).

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of boats of type $j$ assigned to display area $i$.

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

- $I$: All DisplayID in file_0_view_0 (capacity.csv), column DisplayID.
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $c_i$: file_0_view_0, column Capacity, indexed by DisplayID.
- $v_j$: file_1_view_0, column Value, indexed by ProductName.
- $w_j$: file_1_view_0, column Weight, indexed by ProductName.
- $x_{ij}$: Number of boats of type $j$ assigned to display area $i$ (decision variable, nonnegative integer).