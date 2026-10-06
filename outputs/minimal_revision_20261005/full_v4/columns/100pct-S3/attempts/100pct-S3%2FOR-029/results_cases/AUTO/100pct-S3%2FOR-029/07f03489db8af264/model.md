**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_s$: capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)
- $v_p$: value per unit of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: weight per unit of product $p$ (file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq c_s
   \]

2. **Minimum Placement of First Product:**  
   Let $p^*$ be the first product in file_1_view_0 (ProductName, source_row = 0). Then,
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID from file_0_view_0 (capacity.csv), column ShelfID
- $P$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName
- $p^*$: ProductName from file_1_view_0, source_row = 0

---

**Summary:**  
Maximize total value of products allocated to shelves, subject to shelf capacity (by weight), a minimum total placement of the first product, and nonnegative integer variables. All index sets and parameters are mapped directly to the supplied data columns and business identifiers.