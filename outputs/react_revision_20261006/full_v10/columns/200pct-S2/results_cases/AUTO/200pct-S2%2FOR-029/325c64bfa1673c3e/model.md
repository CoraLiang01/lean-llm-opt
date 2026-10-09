Mathematical Model

Index Sets:
- $I$: set of displays (from file_0_view_0, column ShelfID)
- $J$: set of products (from file_1_view_0, column ProductName)

Parameters:
- $c_i$: capacity of display $i$ (from file_0_view_0, column Capacity, indexed by ShelfID)
- $v_j$: value of product $j$ (from file_1_view_0, column Value, indexed by ProductName)
- $w_j$: weight of product $j$ (from file_1_view_0, column Weight, indexed by ProductName)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i \in I$, $j \in J$

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Data Mapping

Index Sets:
- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName

Parameters:
- $c_i$: file_0_view_0, column Capacity, indexed by ShelfID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on display $i$; $i$ from file_0_view_0.ShelfID, $j$ from file_1_view_0.ProductName

Special constraint:
- The "first product" is the product with the smallest source_row in file_1_view_0 (ProductName at source_row 0).