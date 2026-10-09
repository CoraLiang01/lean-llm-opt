Mathematical Model

Index Sets:
- Let S be the set of shelves, with elements indexed by s and identified by ShelfID from file_0_view_0.
- Let P be the set of products, with elements indexed by p and identified by ProductName from file_1_view_0.

Parameters:
- $c_s$: Capacity of shelf $s$ (from file_0_view_0, column Capacity).
- $v_p$: Value per unit of product $p$ (from file_1_view_0, column Value).
- $w_p$: Weight per unit of product $p$ (from file_1_view_0, column Weight).

Decision Variables:
- $x_{sp}$: Number of units of product $p$ to place on shelf $s$.
 Domain: $x_{sp} \in \mathbb{Z}_{\geq 0}$ for all $s \in S$, $p \in P$.

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:

Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s
\]

Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

Data Mapping

Index Sets:
- S (shelves): file_0_view_0, column ShelfID
- P (products): file_1_view_0, column ProductName

Parameters:
- $c_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName

Decision Variables:
- $x_{sp}$: Number of units of product p on shelf s (indexed by ShelfID and ProductName)

All parameters and index sets are defined directly from the returned CSV data, using the exact column and table identifiers.