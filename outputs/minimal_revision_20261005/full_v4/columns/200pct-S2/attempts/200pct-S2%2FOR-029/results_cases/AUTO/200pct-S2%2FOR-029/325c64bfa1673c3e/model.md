**Abstract Mathematical Model**

**Index Sets**
- $S$: set of displays (shelves), indexed by $s$; $s \in S$ corresponds to each ShelfID in file_0_view_0.
- $P$: set of products, indexed by $p$; $p \in P$ corresponds to each ProductName in file_1_view_0.

**Parameters**
- $C_s$: capacity of display $s$ (from Capacity in file_0_view_0, indexed by ShelfID).
- $v_p$: value of product $p$ (from Value in file_1_view_0, indexed by ProductName).
- $w_p$: weight of product $p$ (from Weight in file_1_view_0, indexed by ProductName).
- $p^*$: the first product in file_1_view_0 (ProductName = "Smartphone" from file_1_view_1).

**Decision Variables**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$.

**Objective**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints**
1. **Display Capacity Constraints** (for each display $s$):
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]
2. **First Product Minimum Quantity Constraint**:
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]
3. **Nonnegativity and Integrality**:
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

- $S$: file_0_view_0, column ShelfID
- $P$: file_1_view_0, column ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $p^*$: file_1_view_1, column ProductName (the first product, "Smartphone")