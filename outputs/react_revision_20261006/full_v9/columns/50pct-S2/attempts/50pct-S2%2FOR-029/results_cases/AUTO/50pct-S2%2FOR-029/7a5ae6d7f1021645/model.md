#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business identifier ShelfID from file_0_view_0.
- $J$ = set of products (indexed by $j$), with business identifier ProductName from file_1_view_0.
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable), $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Parameters:
- $v_j$ = value of product $j$ (file_1_view_0, column Value).
- $w_j$ = weight of product $j$ (file_1_view_0, column Weight).
- $C_i$ = capacity of display $i$ (file_0_view_0, column Capacity).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
1. **Display Capacity Constraints** (for all $i \in I$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i
\]

2. **Minimum Quantity of First Product** (let $j^*$ be the ProductName in file_1_view_0, source_row 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]

3. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID.
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $v_j$: file_1_view_0, column Value, keyed by ProductName.
- $w_j$: file_1_view_0, column Weight, keyed by ProductName.
- $C_i$: file_0_view_0, column Capacity, keyed by ShelfID.
- $j^*$: ProductName in file_1_view_0, source_row 0 (the first product listed).

All indices, parameters, and constraints are mapped directly to the corresponding columns and business identifiers in the provided CSV files.