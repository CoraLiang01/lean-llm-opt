**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: Set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from Capacity in file_0_view_0, indexed by ShelfID)
- $v_p$: Value of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: Weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

**Subject to:**

- **Shelf Capacity Constraints:**  
  For each shelf $s \in S$,
  \[
  \sum_{p \in P} w_p \cdot x_{s,p} \leq C_s
  \]

- **Integrality and Nonnegativity:**  
  For all $s \in S$, $p \in P$,
  \[
  x_{s,p} \in \mathbb{Z}_{\geq 0}
  \]

---

**Data Mapping**

- $S$: All ShelfID from `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName from `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, columns `ShelfID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`