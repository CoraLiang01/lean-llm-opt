#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of displays (indexed by $i$), with business identifier ShelfID from file_0_view_0 (capacity.csv)
- $J$: set of products (indexed by $j$), with business identifier ProductName from file_1_view_0 (products.csv)

**Parameters:**
- $c_i$: capacity of display $i$ (file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value of product $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$: weight of product $j$ (file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

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
   Let $j^*$ be the ProductName from file_1_view_0, source_row 0 (the first product in products.csv). Then,
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (Displays): file_0_view_0 (capacity.csv), column ShelfID
- $c_i$: file_0_view_0 (capacity.csv), column Capacity, key ShelfID
- $J$ (Products): file_1_view_0 (products.csv), column ProductName
- $v_j$: file_1_view_0 (products.csv), column Value, key ProductName
- $w_j$: file_1_view_0 (products.csv), column Weight, key ProductName
- $j^*$: file_1_view_0 (products.csv), source_row 0, column ProductName

All indices, parameters, and constraints are mapped directly to the original file and column names as returned by CSVQA.