#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$ = set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.

Parameters:
- $c_i$ = Capacity of display $i$ (from file_0_view_0, column Capacity, keyed by ShelfID).
- $v_j$ = Value of product $j$ (from file_1_view_0, column Value, keyed by ProductName).
- $w_j$ = Weight of product $j$ (from file_1_view_0, column Weight, keyed by ProductName).

Decision variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$.

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. **Display capacity constraints** (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. **Minimum total quantity of the first product** (let $j^*$ be the ProductName in the first row of file_1_view_0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. **Nonnegativity and integrality**:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID.
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $c_i$: file_0_view_0, column Capacity, keyed by ShelfID.
- $v_j$: file_1_view_0, column Value, keyed by ProductName.
- $w_j$: file_1_view_0, column Weight, keyed by ProductName.
- $j^*$: ProductName in file_1_view_0, source_row 0 (the first product listed in products.csv).

All indices, parameters, and constraints are mapped directly to the columns and rows as described above.