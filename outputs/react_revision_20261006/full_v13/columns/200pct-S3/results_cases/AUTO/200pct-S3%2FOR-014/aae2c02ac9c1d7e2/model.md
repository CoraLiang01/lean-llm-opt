#### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$ = set of products, indexed by $p$ (from all ProductName in file_1_view_0)
- $x_{sp}$ = number of units of product $p$ placed on shelf $s$ (decision variable, integer, $\geq 0$)
- $v_p$ = value of product $p$ (from Value in file_1_view_0)
- $w_p$ = weight of product $p$ (from Weight in file_1_view_0)
- $C_s$ = capacity of shelf $s$ (from Capacity in file_0_view_0)

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

**Subject to:**

- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s
\]

- Integer and nonnegativity constraints (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$ (shelves): all ShelfID in table_id=file_0_view_0, column=ShelfID
- $C_s$ (shelf capacity): table_id=file_0_view_0, column=Capacity, keyed by ShelfID
- $P$ (products): all ProductName in table_id=file_1_view_0, column=ProductName
- $v_p$ (product value): table_id=file_1_view_0, column=Value, keyed by ProductName
- $w_p$ (product weight): table_id=file_1_view_0, column=Weight, keyed by ProductName
- $x_{sp}$: integer, $\geq 0$, for all $(s,p) \in S \times P$