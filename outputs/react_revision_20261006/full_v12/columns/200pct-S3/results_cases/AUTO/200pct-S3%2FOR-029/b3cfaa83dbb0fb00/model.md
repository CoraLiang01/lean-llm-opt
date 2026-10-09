### Mathematical Model

**Index Sets:**
- $S$: set of displays (indexed by $s$), from `file_0_view_0.ShelfID`
- $P$: set of products (indexed by $p$), from `file_1_view_0.ProductName$

**Parameters:**
- $C_s$: capacity of display $s$, from `file_0_view_0.Capacity`
- $v_p$: value of product $p$, from `file_1_view_0.Value`
- $w_p$: weight of product $p$, from `file_1_view_0.Weight$

**Decision Variables:**
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Constraints:**
1. **Display Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
   \]
2. **Minimum Quantity for First Product:**
   \[
   \sum_{s \in S} x_{s p^*} \geq 5
   \]
   where $p^*$ is the first product in `file_1_view_0.ProductName` (i.e., the product in source row 0).

3. **Nonnegativity and Integrality:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

### Data Mapping

- $S$: `file_0_view_0.ShelfID`
- $P$: `file_1_view_0.ProductName`
- $C_s$: `file_0_view_0.Capacity` (for each $s$)
- $v_p$: `file_1_view_0.Value` (for each $p$)
- $w_p$: `file_1_view_0.Weight` (for each $p$)
- $p^*$: product with `file_1_view_0.source_row = 0` (i.e., the first product listed)

All indices, parameters, and constraints are mapped directly to the original CSV columns and rows as specified.