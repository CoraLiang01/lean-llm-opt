**Abstract Mathematical Model**

**Sets:**
- $S$: Set of shelves (indexed by $s$), from capacity.csv, column ShelfID.
- $P$: Set of products (indexed by $p$), from products.csv, column ProductName.

**Parameters:**
- $c_s$: Capacity of shelf $s$ (capacity.csv, column Capacity).
- $v_p$: Value of product $p$ (products.csv, column Value).
- $w_p$: Weight of product $p$ (products.csv, column Weight).

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$.

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

2. **Minimum Allocation of First Product:**  
   Let $p^*$ denote the first product in products.csv (ProductName = "Smartphone"):
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: capacity.csv, column ShelfID
- $P$: products.csv, column ProductName
- $c_s$: capacity.csv, column Capacity (keyed by ShelfID)
- $v_p$: products.csv, column Value (keyed by ProductName)
- $w_p$: products.csv, column Weight (keyed by ProductName)
- $x_{sp}$: Decision variable for allocation of product $p$ to shelf $s$