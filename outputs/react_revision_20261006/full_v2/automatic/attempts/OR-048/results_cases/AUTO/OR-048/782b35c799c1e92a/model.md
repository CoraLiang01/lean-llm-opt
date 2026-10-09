Mathematical Model

Sets:
- $I$: set of storage areas (indexed by $i$), with StorageID from file_0_view_0.
- $J$: set of air conditioner types (indexed by $j$), with ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of storage area $i$ (from file_0_view_0, column Capacity).
- $v_j$: value of air conditioner type $j$ (from file_1_view_0, column Value).
- $w_j$: size (weight) of air conditioner type $j$ (from file_1_view_0, column Weight).

Decision Variables:
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$.
 Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$.

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

- $I$: StorageID from file_0_view_0
- $J$: ProductName from file_1_view_0
- $c_i$: file_0_view_0, column Capacity, key StorageID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: number of units of product $j$ in storage area $i$ (decision variable, indexed by StorageID and ProductName)