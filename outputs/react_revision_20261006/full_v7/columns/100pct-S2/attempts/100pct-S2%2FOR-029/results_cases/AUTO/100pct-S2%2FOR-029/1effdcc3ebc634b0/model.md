#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business identifier ShelfID from file_0_view_0.
- $J$ = set of products (indexed by $j$), with business identifier ProductName from file_1_view_0.

Parameters:
- $c_i$ = capacity of display $i$ (Capacity from file_0_view_0, indexed by ShelfID)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. **Display capacity constraints** (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. **Minimum allocation for the first product** (let $j^*$ be the ProductName in file_1_view_0 with source_row = 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. **Nonnegativity and integrality** (for all $i \in I$, $j \in J$):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, indexed by ShelfID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName
- $j^*$: ProductName in file_1_view_0, source_row = 0 (the first product listed)

All variables, indices, and parameters are defined strictly according to the returned data and user query.