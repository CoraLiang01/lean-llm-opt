Abstract Mathematical Model

Index Sets:
- $I$: Set of storage areas (indexed by $i$), from file_0_view_0, column StorageID.
- $J$: Set of air conditioner types (indexed by $j$), from file_1_view_0, column ProductName.

Parameters:
- $c_i$: Capacity of storage area $i$, from file_0_view_0, column Capacity.
- $v_j$: Value of air conditioner type $j$, from file_1_view_0, column Value.
- $w_j$: Size (weight) of air conditioner type $j$, from file_1_view_0, column Weight.

Decision Variables:
- $x_{ij}$: Number of units of air conditioner type $j$ to place in storage area $i$.
  - Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

Subject to:

Capacity constraints (for each storage area):
\[
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i, \quad \forall i \in I
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
\]

---

Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $c_i$: file_0_view_0, columns StorageID (key), Capacity (value)
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, columns ProductName (key), Value (value)
- $w_j$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{ij}$: Decision variable for each $(i, j)$ pair

All parameters and sets are defined directly from the returned rows and columns of the source files, preserving original order and identifiers.