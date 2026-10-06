**Mathematical Optimization Model**

---

**Index Sets:**

- $S$: Set of shelves (indexed by $s$), from `file_0_view_0.ShelfID`
- $P$: Set of products (indexed by $p$), from `file_1_view_0.ProductName$

---

**Parameters:**

- $c_s$: Capacity of shelf $s$ (from `file_0_view_0.Capacity`)
- $v_p$: Value of product $p$ (from `file_1_view_0.Value`)
- $w_p$: Weight of product $p$ (from `file_1_view_0.Weight`)

---

**Decision Variables:**

- $x_{sp}$: Number of units of product $p$ placed on shelf $s$  
  Domain: $x_{sp} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), $\forall s \in S, p \in P$

---

**Objective:**

\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

---

**Constraints:**

1. **Shelf Capacity Constraints:**  
   For each shelf $s \in S$,
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq c_s
   \]

2. **Minimum Placement of First Product:**  
   Let $p^*$ be the first product in the order of `file_1_view_0` (i.e., the product with the smallest `source_row`, which is `ProductName` = "Smartphone"):
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in S, p \in P
   \]

---

**Data Mapping**

- $S$: All `ShelfID` in `file_0_view_0`
- $P$: All `ProductName` in `file_1_view_0`
- $c_s$: `file_0_view_0.Capacity` (matched by `ShelfID`)
- $v_p$: `file_1_view_0.Value` (matched by `ProductName`)
- $w_p$: `file_1_view_0.Weight` (matched by `ProductName`)
- $p^*$: The `ProductName` in `file_1_view_0` with `source_row = 0` ("Smartphone")

---

**Summary:**  
Maximize total value of products allocated to shelves, subject to shelf capacity (by weight), nonnegativity/integrality, and a minimum total allocation of the first product across all shelves. All index sets and parameters are mapped directly from the supplied CSV data using the specified columns and order.