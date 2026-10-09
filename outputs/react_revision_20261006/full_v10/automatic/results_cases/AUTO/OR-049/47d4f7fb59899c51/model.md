Mathematical Model

Index Sets:
- Let $S$ be the set of ShelfIDs from file_0_view_0 (capacity.csv).
- Let $P$ be the set of ProductName from file_1_view_0 (products.csv).

Parameters:
- $c_s$: Capacity of shelf $s \in S$ (from file_0_view_0, column Capacity, key ShelfID).
- $v_p$: Value of product $p \in P$ (from file_1_view_0, column Value, key ProductName).
- $w_p$: Weight of product $p \in P$ (from file_1_view_0, column Weight, key ProductName).

Decision Variables:
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$, $x_{sp} \in \mathbb{Z}_{\geq 0}$ for all $s \in S$, $p \in P$.

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:

Shelf Capacity Constraints:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \quad \forall s \in S
\]

Integrality and Nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: file_0_view_0, column ShelfID
- $P$: file_1_view_0, column ProductName
- $c_s$: file_0_view_0, columns ShelfID (key), Capacity (value)
- $v_p$: file_1_view_0, columns ProductName (key), Value (value)
- $w_p$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{sp}$: Decision variable for each $(s,p) \in S \times P$ as defined above

All parameters and index sets are derived directly from the current CSV data as specified.