ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of displays (from file_0_view_0.ShelfID)
- $J$: set of products (from file_1_view_0.ProductName)

Parameters:
- $c_i$: capacity of display $i$ (from file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value per unit of product $j$ (from file_1_view_0, column Value, key ProductName)
- $w_j$: weight per unit of product $j$ (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

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

DATA MAPPING

- $I$: file_0_view_0.ShelfID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $j^*$: file_1_view_0.ProductName with source_row = 0
- $x_{ij}$: number of units of product $j$ placed on display $i$