#### Abstract Mathematical Model

**Sets:**
- $I$: set of displays (indexed by $i$), from file_0_view_0.ShelfID
- $J$: set of products (indexed by $j$), from file_1_view_0.ProductName

**Parameters:**
- $c_i$: capacity of display $i$, from file_0_view_0.Capacity
- $v_j$: value of product $j$, from file_1_view_0.Value
- $w_j$: weight of product $j$, from file_1_view_0.Weight

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
2. **Minimum Allocation of First Product:**  
   Let $j^*$ be the first product in file_1_view_0 (ProductName = "Smartphone"):
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
- $j^*$: file_1_view_0.ProductName = "Smartphone" (source_row = 0)

All parameters and indices are to be used exactly as provided in the source files and rows.