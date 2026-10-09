#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$ = set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable), $x_{ij} \in \mathbb{Z}_{\geq 0}$.

Parameters:
- $c_i$ = capacity of display $i$ (from Capacity in file_0_view_0, indexed by ShelfID).
- $v_j$ = value of product $j$ (from Value in file_1_view_0, indexed by ProductName).
- $w_j$ = weight of product $j$ (from Weight in file_1_view_0, indexed by ProductName).

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

Subject to:
1. **Display Capacity Constraints** (for all $i \in I$):
\[
\sum_{j \in J} w_j x_{ij} \leq c_i
\]
2. **Minimum Quantity of First Product** (let $j^*$ be the ProductName in file_1_view_0 with source_row = 0):
\[
\sum_{i \in I} x_{i j^*} \geq 5
\]
3. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column "ShelfID".
- $J$: All ProductName in file_1_view_0 (products.csv), column "ProductName".
- $c_i$: file_0_view_0, columns "ShelfID" (key), "Capacity" (value).
- $v_j$: file_1_view_0, columns "ProductName" (key), "Value" (value).
- $w_j$: file_1_view_0, columns "ProductName" (key), "Weight" (value).
- $j^*$: ProductName in file_1_view_0 with source_row = 0 (the first product listed).

- Decision variables $x_{ij}$: Number of units of product $j$ placed on display $i$, for all $i \in I$, $j \in J$.