---
#### Abstract Mathematical Model

**Sets:**
- $I$: set of displays (indexed by $i$), corresponding to ShelfID in file_0_view_0 (capacity.csv)
- $J$: set of products (indexed by $j$), corresponding to ProductName in file_1_view_0 (products.csv)

**Parameters:**
- $c_i$: capacity of display $i$ (Capacity column in file_0_view_0)
- $v_j$: value of product $j$ (Value column in file_1_view_0)
- $w_j$: weight of product $j$ (Weight column in file_1_view_0)

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
   Let $j^*$ be the product corresponding to source_row = 0 in file_1_view_0 (i.e., the first product in products.csv, ProductName = "Smartphone"):
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (Displays): file_0_view_0.ShelfID
- $c_i$: file_0_view_0.Capacity, keyed by ShelfID
- $J$ (Products): file_1_view_0.ProductName
- $v_j$: file_1_view_0.Value, keyed by ProductName
- $w_j$: file_1_view_0.Weight, keyed by ProductName
- $j^*$: file_1_view_0 source_row = 0 (ProductName = "Smartphone")

All indices, parameters, and constraints are mapped directly to the original columns and row order as returned.