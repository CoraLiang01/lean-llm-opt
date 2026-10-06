## Abstract Mathematical Model

**Sets:**
- $I$: set of displays (indexed by $i$), with business key ShelfID from capacity.csv
- $J$: set of products (indexed by $j$), with business key ProductName from products.csv

**Parameters:**
- $c_i$: capacity of display $i$ (from Capacity column in capacity.csv)
- $v_j$: value of product $j$ (from Value column in products.csv)
- $w_j$: weight of product $j$ (from Weight column in products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

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
   Let $j^*$ be the ProductName of the first row in products.csv. Then,
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- $I$ (displays): ShelfID from file_0_view_0 (capacity.csv)
- $J$ (products): ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column in file_0_view_0, indexed by ShelfID
- $v_j$: Value column in file_1_view_0, indexed by ProductName
- $w_j$: Weight column in file_1_view_0, indexed by ProductName
- $j^*$: ProductName from source_row 0 of file_1_view_0 (products.csv)

All indices, parameters, and constraints are mapped directly to the original file columns and row order as required.