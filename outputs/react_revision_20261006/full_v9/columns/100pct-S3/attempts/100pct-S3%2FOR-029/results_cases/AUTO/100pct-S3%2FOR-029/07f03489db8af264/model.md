#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business key $ShelfID$ from file_0_view_0.
- $J$ = set of products (indexed by $j$), with business key $ProductName$ from file_1_view_0.
- $v_j$ = value of product $j$ ($Value$ from file_1_view_0).
- $w_j$ = weight of product $j$ ($Weight$ from file_1_view_0).
- $C_i$ = capacity of display $i$ ($Capacity$ from file_0_view_0).
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer).

Let $j^*$ denote the business key of the first product in file_1_view_0 (i.e., the product with source_row = 0).

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**
1. **Display capacity constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
   \]
2. **Minimum total quantity of the first product:**
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: All $ShelfID$ in file_0_view_0 (capacity.csv), in source order.
- $J$: All $ProductName$ in file_1_view_0 (products.csv), in source order.
- $v_j$: $Value$ column in file_1_view_0, keyed by $ProductName$.
- $w_j$: $Weight$ column in file_1_view_0, keyed by $ProductName$.
- $C_i$: $Capacity$ column in file_0_view_0, keyed by $ShelfID$.
- $j^*$: The $ProductName$ of the first row (source_row = 0) in file_1_view_0.
- $x_{ij}$: Number of units of product $j$ placed on display $i$.

All indices, parameters, and constraints are mapped directly to the corresponding columns and business keys as described above.