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
2. **Minimum Placement of First Product** (let $p^*$ be the product in the first row of file_1_view_0, i.e., $p^* =$ "Smartphone"):
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]
3. **Nonnegativity and Integrality**:
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: All ShelfID in `file_0_view_0` (capacity.csv), column `ShelfID`
- $P$: All ProductName in `file_1_view_0` (products.csv), column `ProductName`
- $C_s$: `file_0_view_0`, column `Capacity`, indexed by `ShelfID`
- $v_p$: `file_1_view_0`, column `Value`, indexed by `ProductName`
- $w_p$: `file_1_view_0`, column `Weight`, indexed by `ProductName`
- $p^*$: First `ProductName` in `file_1_view_0` (row 0): "Smartphone"

---

**Summary:**  
Maximize total value of products allocated to displays, subject to each display's capacity, with at least 5 units of the first product ("Smartphone") placed in total, and all variables nonnegative integers. All index sets and parameters are mapped directly to the supplied data columns and rows.