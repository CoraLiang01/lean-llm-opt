### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$ = set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.
- $c_i$ = capacity of display $i$, from Capacity in file_0_view_0.
- $v_j$ = value of product $j$, from Value in file_1_view_0.
- $w_j$ = weight of product $j$, from Weight in file_1_view_0.
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer).

Let $j^*$ denote the index of the first product in file_1_view_0 (ProductName = "Smartphone").

#### Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

#### Constraints:
1. **Display Capacity Constraints** (for each display $i$):
   \[
   \sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I
   \]

2. **Minimum Quantity of First Product**:
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]

3. **Nonnegativity and Integrality**:
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column "ShelfID"
- $J$: All ProductName in file_1_view_0 (products.csv), column "ProductName"
- $c_i$: file_0_view_0, column "Capacity", keyed by "ShelfID"
- $v_j$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_j$: file_1_view_0, column "Weight", keyed by "ProductName"
- $j^*$: ProductName in file_1_view_0, source_row = 0 ("Smartphone")
- $x_{ij}$: Number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer)

All index sets, parameters, and constraints are mapped directly to the columns and rows as described above.