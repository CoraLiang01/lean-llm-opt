Mathematical Model

Index Sets:
- Let $I$ be the set of ShelfIDs from file_0_view_0 (capacity.csv).
- Let $J$ be the set of ProductNames from file_1_view_0 (products.csv).

Parameters:
- $c_i$: Capacity of shelf $i \in I$ (file_0_view_0, column Capacity).
- $v_j$: Value of product $j \in J$ (file_1_view_0, column Value).
- $w_j$: Weight of product $j \in J$ (file_1_view_0, column Weight).

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$, for all $i \in I$, $j \in J$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Shelf capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]

2. Minimum allocation of the first product (let $j^*$ be the ProductName in file_1_view_0 with source_row = 0):
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
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName

Special index:
- $j^*$: file_1_view_0, source_row = 0, column ProductName

Decision Variables:
- $x_{ij}$: for all $i \in I$, $j \in J$ (number of units of product $j$ on shelf $i$)

All parameters and index sets are mapped directly from the specified columns and rows in the returned CSV data.