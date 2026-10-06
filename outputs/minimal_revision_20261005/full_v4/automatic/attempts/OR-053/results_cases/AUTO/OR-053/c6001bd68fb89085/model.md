**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from Capacity in file_0_view_0, keyed by ShelfID)
- $v_p$: Value of product $p$ (from Value in file_1_view_0, keyed by ProductName)
- $w_p$: Weight of product $p$ (from Weight in file_1_view_0, keyed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID from file_0_view_0
- $P$: All ProductName from file_1_view_0
- $C_s$: file_0_view_0, column "Capacity", keyed by "ShelfID"
- $v_p$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_p$: file_1_view_0, column "Weight", keyed by "ProductName"
- $x_{sp}$: Decision variable for each $(s,p)$ pair

---

**Summary:**  
This model determines the integer number of units of each product to allocate to each shelf to maximize total value, subject to each shelf's capacity. All data and index sets are mapped directly from the provided CSV files and columns.