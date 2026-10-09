### Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), corresponding to all ShelfID in file_0_view_0.
- $J$: set of products (indexed by $j$), corresponding to all ProductName in file_1_view_0.

**Parameters:**
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID).
- $v_j$: value per unit of product $j$ (file_1_view_0, column Value, key ProductName).
- $w_j$: weight per unit of product $j$ (file_1_view_0, column Weight, key ProductName).

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j x_{ij} \leq c_i
   \]
2. **Minimum Quantity of First Product:**  
   Let $j^*$ be the ProductName in the first row of file_1_view_0. Then,
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID.
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $c_i$: file_0_view_0, column Capacity, key ShelfID.
- $v_j$: file_1_view_0, column Value, key ProductName.
- $w_j$: file_1_view_0, column Weight, key ProductName.
- $j^*$: ProductName in file_1_view_0, source_row 0.

All variables, parameters, and constraints are indexed and mapped directly to the original data columns and business identifiers as required.