#### Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to all ShelfID values in file_0_view_0.
- $J$: set of products (indexed by $j$), corresponding to all ProductName values in file_1_view_0.

**Parameters:**
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, keyed by ShelfID).
- $v_j$: value per unit of product $j$ (file_1_view_0, column Value, keyed by ProductName).
- $w_j$: weight per unit of product $j$ (file_1_view_0, column Weight, keyed by ProductName).

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ placed on display $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i \in I$, $j \in J$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i
   \]
2. **Minimum Quantity of First Product:**  
   Let $j^*$ be the ProductName in file_1_view_0 with source_row = 0 (the first product in products.csv). Then,
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
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
- $j^*$: ProductName in file_1_view_0, source_row = 0 (the first product listed in products.csv).

---

**All index sets, parameters, and constraints are defined directly from the supplied data and user requirements.**