**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of displays (shelves), indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: capacity of display $s$ (from Capacity in file_0_view_0, indexed by ShelfID)
- $v_p$: value of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**

1. **Display Capacity Constraints:**  
   For each display $s \in S$,
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s
   \]

2. **Minimum Placement of First Product:**  
   Let $p^*$ be the first product in file_1_view_0 (ProductName = "Smartphone"):
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**  
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID from `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName from `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, columns `ShelfID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`
- $p^*$: First ProductName in `file_1_view_0` (source_row = 0)

---

**Summary:**  
Maximize total value of products allocated to displays, subject to each display's capacity and a minimum total allocation of the first product, with all variables nonnegative integers. All index sets and parameters are mapped directly to the supplied data columns and business identifiers.