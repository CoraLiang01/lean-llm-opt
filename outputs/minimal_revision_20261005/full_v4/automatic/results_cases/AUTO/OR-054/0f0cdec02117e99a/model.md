**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: capacity of shelf $s$ (from Capacity in file_0_view_0, keyed by ShelfID)
- $v_p$: value per unit of product $p$ (from Value in file_1_view_0, keyed by ProductName)
- $w_p$: weight (space requirement) per unit of product $p$ (from Weight in file_1_view_0, keyed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

---

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

2. **Nonnegativity and Integrality:**  
   For all $s \in S$, $p \in P$,
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $S$: All ShelfID from `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName from `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, columns `ShelfID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$

---

**Summary:**  
Maximize total value of products allocated to shelves, subject to each shelf's capacity, using nonnegative integer allocations per product per shelf. All parameters and index sets are mapped directly from the supplied CSV data.