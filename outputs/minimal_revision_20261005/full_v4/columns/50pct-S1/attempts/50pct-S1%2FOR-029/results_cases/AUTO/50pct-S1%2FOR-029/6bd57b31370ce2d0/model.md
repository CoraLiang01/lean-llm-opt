**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of displays (shelves), indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters:**
- $C_s$: capacity of display $s$ (from Capacity in file_0_view_0, indexed by ShelfID)
- $v_p$: value of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName)
- $p^*$: the first product in file_1_view_0 (ProductName = "Smartphone" from file_1_view_1)

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

1. **Display Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **First Product Minimum Allocation:**
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]

3. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID in `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName in `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, columns `ShelfID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`
- $p^*$: `file_1_view_1`, column `ProductName` (the first product, "Smartphone")