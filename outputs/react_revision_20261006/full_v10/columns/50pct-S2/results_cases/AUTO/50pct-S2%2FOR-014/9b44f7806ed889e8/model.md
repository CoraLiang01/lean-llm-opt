ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $S$ be the set of shelves, indexed by $s$, with ShelfID from file_0_view_0.
- Let $P$ be the set of products, indexed by $p$, with ProductName from file_1_view_0.

Parameters:
- $c_s$: Capacity of shelf $s$ (file_0_view_0, column Capacity, key ShelfID)
- $v_p$: Value of product $p$ (file_1_view_0, column Value, key ProductName)
- $w_p$: Weight of product $p$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: Number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
- Shelf capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{s,p} \leq c_s
\]
- Integrality and nonnegativity (for all $s \in S$, $p \in P$):
\[
x_{s,p} \in \mathbb{Z}_{\geq 0}
\]

DATA MAPPING

Index Sets:
- $S$: file_0_view_0, column ShelfID
- $P$: file_1_view_0, column ProductName

Parameters:
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

Decision Variables:
- $x_{s,p}$: Number of units of product $p$ to place on shelf $s$ (indexed by file_0_view_0.ShelfID and file_1_view_0.ProductName)

All constraints and parameters are mapped directly to the corresponding columns and keys in the provided CSV files.