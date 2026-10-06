**Sets:**

- $S$ = set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$ = set of products, indexed by $p$ (from file_1_view_0, column ProductName)

**Parameters:**

- $C_s$ = capacity of shelf $s$ (from file_0_view_0, column Capacity)
- $v_p$ = value of product $p$ (from file_1_view_0, column Value)
- $w_p$ = weight of product $p$ (from file_1_view_0, column Weight)

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
   \sum_{p \in P} w_p \, x_{sp} \leq C_s
   \]

2. **Minimum Allocation of First Product:**  
   Let $p^*$ be the first product in file_1_view_0 (ProductName = "Smartphone"):
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

### Data Mapping

- $S$: file_0_view_0, column ShelfID
- $P$: file_1_view_0, column ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $p^*$: file_1_view_0, ProductName in source_row 0 ("Smartphone")