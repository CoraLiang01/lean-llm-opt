#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$ = set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.
- $c_i$ = capacity of display $i$ (parameter: Capacity from file_0_view_0, indexed by ShelfID).
- $v_j$ = value of product $j$ (parameter: Value from file_1_view_0, indexed by ProductName).
- $w_j$ = weight of product $j$ (parameter: Weight from file_1_view_0, indexed by ProductName).
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer).

Let $j^*$ denote the ProductName in file_1_view_0 with source_row = 0 (the first product).

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**
1. **Display capacity constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
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

- $I$: All ShelfID in table_id file_0_view_0, column ShelfID.
- $J$: All ProductName in table_id file_1_view_0, column ProductName.
- $c_i$: Capacity from file_0_view_0, column Capacity, indexed by ShelfID.
- $v_j$: Value from file_1_view_0, column Value, indexed by ProductName.
- $w_j$: Weight from file_1_view_0, column Weight, indexed by ProductName.
- $j^*$: ProductName from file_1_view_0, source_row = 0.
- $x_{ij}$: Decision variable for units of product $j$ on display $i$.

All index sets, parameters, and constraints are defined using the exact columns and business identifiers from the returned data.