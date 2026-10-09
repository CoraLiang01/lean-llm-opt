Mathematical Model

Index Sets:
- $I$: set of displays (ShelfID from file_0_view_0)
- $J$: set of products (ProductName from file_1_view_0)

Parameters:
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value of product $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$: weight of product $j$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (for $j^* =$ first ProductName in file_1_view_0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Data Mapping

Index Sets:
- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName

Parameters:
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName

Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$ (indexed by ShelfID and ProductName)

Special constraint:
- $j^*$ is the ProductName in file_1_view_0 with source_row = 0 (the first product listed).