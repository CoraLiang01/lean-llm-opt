**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves (indexed by $s$), from all ShelfID in file_0_view_0.
- $P$: set of products (indexed by $p$), from all ProductName in file_1_view_0.

**Parameters:**
- $C_s$: capacity of shelf $s$ (from Capacity in file_0_view_0, indexed by ShelfID).
- $v_p$: value of product $p$ (from Value in file_1_view_0, indexed by ProductName).
- $w_p$: weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName).

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$.

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{s,p} \leq C_s
   \]

2. **Minimum Placement of First Product:**  
   Let $p^*$ be the ProductName in file_1_view_0 with source_row = 0 (i.e., the first product in the file).
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID.
- $P$: All ProductName in file_1_view_0 (products.csv), column ProductName.
- $C_s$: file_0_view_0, columns ShelfID and Capacity.
- $v_p$: file_1_view_0, columns ProductName and Value.
- $w_p$: file_1_view_0, columns ProductName and Weight.
- $p^*$: ProductName in file_1_view_0, source_row = 0.

---

**Summary:**  
Maximize total value of products allocated to shelves, subject to shelf capacity (weight) constraints, a minimum total allocation of the first product, and nonnegative integer variables for each product-shelf pair. All index sets and parameters are mapped directly to the provided data columns and rows.