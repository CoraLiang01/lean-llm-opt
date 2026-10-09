Mathematical Model

Sets:
- $I$: set of shelves (indexed by $i$), from file_0_view_0, column ShelfID
- $J$: set of products (indexed by $j$), from file_1_view_0, column ProductName

Parameters:
- $c_i$: capacity of shelf $i$ (file_0_view_0, column Capacity)
- $v_j$: value of product $j$ (file_1_view_0, column Value)
- $w_j$: weight of product $j$ (file_1_view_0, column Weight)

Decision Variables:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Shelf capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum allocation of the first product (let $j^*$ be the product in file_1_view_0, source_row = 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I, j \in J
\]

Data Mapping

- $I$: file_0_view_0, column ShelfID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $j^*$: file_1_view_0, source_row = 0 (ProductName = "Smartphone")