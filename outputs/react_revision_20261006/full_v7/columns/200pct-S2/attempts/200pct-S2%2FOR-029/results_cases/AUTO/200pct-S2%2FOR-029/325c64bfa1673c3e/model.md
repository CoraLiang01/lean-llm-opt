#### Mathematical Model

Let:
- $I$ = set of displays (indexed by $i$), with elements given by the "ShelfID" column in capacity.csv.
- $J$ = set of products (indexed by $j$), with elements given by the "ProductName" column in products.csv.
- $v_j$ = value of product $j$, from the "Value" column in products.csv.
- $w_j$ = weight of product $j$, from the "Weight" column in products.csv.
- $C_i$ = capacity of display $i$, from the "Capacity" column in capacity.csv.
- $x_{ij}$ = number of units of product $j$ placed on display $i$ (decision variable, integer, $\geq 0$).

Let $j^*$ denote the product with ProductName = "Smartphone" (the first product in products.csv).

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

**Subject to:**

1. **Display Capacity Constraints:**
   \[
   \sum_{j \in J} w_j x_{ij} \leq C_i \quad \forall i \in I
   \]

2. **Minimum Quantity of First Product:**
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: All "ShelfID" values from capacity.csv (table_id: file_0_view_0, column: ShelfID)
- $J$: All "ProductName" values from products.csv (table_id: file_1_view_0, column: ProductName)
- $v_j$: "Value" from products.csv (table_id: file_1_view_0, column: Value, keyed by ProductName)
- $w_j$: "Weight" from products.csv (table_id: file_1_view_0, column: Weight, keyed by ProductName)
- $C_i$: "Capacity" from capacity.csv (table_id: file_0_view_0, column: Capacity, keyed by ShelfID)
- $j^*$: ProductName = "Smartphone" (table_id: file_1_view_1, column: ProductName)
- $x_{ij}$: Decision variable for units of product $j$ on display $i$ (indexed by $I \times J$)