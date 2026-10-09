Mathematical Optimization Model

Index Sets:
- $I$: set of displays (shelves), indexed by $i$, with business identifier ShelfID from file_0_view_0.
- $J$: set of products, indexed by $j$, with business identifier ProductName from file_1_view_0.

Parameters:
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID).
- $v_j$: value per unit of product $j$ (file_1_view_0, column Value, key ProductName).
- $w_j$: weight per unit of product $j$ (file_1_view_0, column Weight, key ProductName).

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i \in I$, $j \in J$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (for $j^* =$ first ProductName in file_1_view_0, i.e., source_row 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

Data Mapping:
- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $j^*$: ProductName from file_1_view_0, source_row 0
- $x_{ij}$: number of units of product $j$ placed on display $i$ (decision variable, indexed by $I \times J$)