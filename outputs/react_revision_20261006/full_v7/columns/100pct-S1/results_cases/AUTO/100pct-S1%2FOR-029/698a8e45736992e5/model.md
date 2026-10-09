#### Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to all distinct values of ShelfID in file_0_view_0.
- $J$: set of products (indexed by $j$), corresponding to all distinct values of ProductName in file_1_view_0.

**Parameters:**
- $c_i$: capacity of display $i$ (from Capacity in file_0_view_0, indexed by ShelfID).
- $v_j$: value per unit of product $j$ (from Value in file_1_view_0, indexed by ProductName).
- $w_j$: weight per unit of product $j$ (from Weight in file_1_view_0, indexed by ProductName).

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$.

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
   Let $j^*$ be the product corresponding to the first row of file_1_view_0 (i.e., ProductName in source_row 0). Then,
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: All ShelfID values from file_0_view_0 (capacity.csv), in original row order.
- $J$: All ProductName values from file_1_view_0 (products.csv), in original row order.
- $c_i$: Capacity column from file_0_view_0, indexed by ShelfID.
- $v_j$: Value column from file_1_view_0, indexed by ProductName.
- $w_j$: Weight column from file_1_view_0, indexed by ProductName.
- $j^*$: ProductName from source_row 0 of file_1_view_0 (products.csv).

All index sets, parameters, and constraints are defined using the exact columns and business identifiers from the provided data.