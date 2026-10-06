**Abstract Mathematical Model**

**Index Sets**
- $S$: set of displays (shelves), indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$: set of products, indexed by $p$ (from all ProductName in file_1_view_0)

**Parameters**
- $C_s$: capacity of display $s$ (from Capacity in file_0_view_0, indexed by ShelfID)
- $v_p$: value of product $p$ (from Value in file_1_view_0, indexed by ProductName)
- $w_p$: weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$

**Objective**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints**
1. **Display Capacity Constraints** (for each $s \in S$):
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s
   \]
2. **First Product Minimum Allocation** (for $p^*$ = "Smartphone"):
   \[
   \sum_{s \in S} x_{s,p^*} \geq 5
   \]
3. **Nonnegativity and Integrality**:
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID in `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName in `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, columns `ShelfID`, `Capacity`
- $v_p$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_p$: `file_1_view_0`, columns `ProductName`, `Weight`
- $p^*$: "Smartphone" (first product in `file_1_view_1`, products.csv)

---

**Notes**
- All index sets and parameters are defined directly from the returned data, preserving original order and identifiers.
- All constraints and variable domains are as specified in the query and data.