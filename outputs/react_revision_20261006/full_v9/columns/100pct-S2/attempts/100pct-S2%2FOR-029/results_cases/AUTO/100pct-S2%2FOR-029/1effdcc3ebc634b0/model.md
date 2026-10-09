#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with business identifier ShelfID from file_0_view_0.
- $J$ = set of products (indexed by $j$), with business identifier ProductName from file_1_view_0.
- $v_j$ = value of product $j$ (Value column, file_1_view_0).
- $w_j$ = weight of product $j$ (Weight column, file_1_view_0).
- $C_i$ = capacity of display $i$ (Capacity column, file_0_view_0).
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable).

$\text{Maximize} \quad \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

Subject to:
1. Display capacity constraints:
   $$
   \sum_{j \in J} w_j x_{ij} \leq C_i \qquad \forall i \in I
   $$
2. Minimum allocation for the first product (the product in the first row of file_1_view_0, i.e., ProductName with source_row = 0):
   $$
   \sum_{i \in I} x_{i j^*} \geq 5
   $$
   where $j^*$ is the ProductName in source_row = 0 of file_1_view_0.
3. Nonnegativity and integrality:
   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$: All ShelfID values from file_0_view_0 (capacity.csv), in source order.
- $J$: All ProductName values from file_1_view_0 (products.csv), in source order.
- $v_j$: file_1_view_0, column Value, keyed by ProductName.
- $w_j$: file_1_view_0, column Weight, keyed by ProductName.
- $C_i$: file_0_view_0, column Capacity, keyed by ShelfID.
- $x_{ij}$: Number of units of product $j$ placed on display $i$ (decision variable, nonnegative integer).
- $j^*$: ProductName in source_row = 0 of file_1_view_0 (the first product listed in products.csv).