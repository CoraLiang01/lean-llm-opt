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
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Subject to:**

- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s
\]

- Nonnegativity and integrality (for all $s \in S$, $p \in P$):
\[
x_{sp} \in \mathbb{Z}_{\geq 0}
\]

---

#### Data Mapping

- $S$: All ShelfID from file_0_view_0 (capacity.csv), column ShelfID
- $P$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$