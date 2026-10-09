#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business IDs $i \in$ ShelfID from file_0_view_0
- $J$ = set of products (indexed by $j$), with business IDs $j \in$ ProductName from file_1_view_0

Parameters:
- $c_i$ = capacity of display $i$ (Capacity from file_0_view_0)
- $v_j$ = value of product $j$ (Value from file_1_view_0)
- $w_j$ = weight of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ placed on display $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. Display capacity constraints (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. Minimum total quantity of the first product (for $j^* =$ first ProductName in file_1_view_0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $j^*$: The ProductName in file_1_view_0, source_row 0 (the first product listed)

All indices, parameters, and constraints are mapped directly to the original data columns and business IDs.