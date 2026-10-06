**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves (indexed by $s$), with business identifier ShelfID from file_0_view_0.
- $P$: set of products (indexed by $p$), with business identifier ProductName from file_1_view_0.

**Parameters:**
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column Capacity).
- $v_p$: value of product $p$ (from file_1_view_0, column Value).
- $w_p$: weight of product $p$ (from file_1_view_0, column Weight).

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
   \sum_{p \in P} w_p \, x_{s,p} \leq c_s
   \]

2. **Minimum Placement of First Product:**  
   Let $p^*$ be the product in the first row of file_1_view_0 (i.e., ProductName = "Smartphone"):
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{s,p} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID values from file_0_view_0, column ShelfID.
- $P$: All ProductName values from file_1_view_0, column ProductName.
- $c_s$: file_0_view_0, column Capacity, keyed by ShelfID.
- $v_p$: file_1_view_0, column Value, keyed by ProductName.
- $w_p$: file_1_view_0, column Weight, keyed by ProductName.
- $p^*$: ProductName in file_1_view_0, source_row = 0.

---

**Summary:**  
Maximize total value of products placed on shelves, subject to shelf capacity (weight) constraints, a minimum total placement of the first product, and nonnegative integer variables for each product-shelf combination. All index sets and parameters are mapped directly to the provided data columns and business identifiers.