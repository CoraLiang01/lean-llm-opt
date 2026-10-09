Mathematical Model

Index Sets:
- $S$: set of displays (from file_0_view_0.ShelfID)
- $P$: set of products (from file_1_view_0.ProductName)

Parameters:
- $C_s$: capacity of display $s$ (from file_0_view_0, column Capacity, key ShelfID)
- $v_p$: value of product $p$ (from file_1_view_0, column Value, key ProductName)
- $w_p$: weight of product $p$ (from file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on display $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:

1. Display capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s
\]

2. Minimum allocation of the first product (let $p^*$ be the product with the smallest source_row in file_1_view_0):
\[
\sum_{s \in S} x_{s p^*} \geq 5
\]

3. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: file_0_view_0.ShelfID
- $P$: file_1_view_0.ProductName
- $C_s$: file_0_view_0.Capacity, keyed by ShelfID
- $v_p$: file_1_view_0.Value, keyed by ProductName
- $w_p$: file_1_view_0.Weight, keyed by ProductName
- $p^*$: file_1_view_0.ProductName with source_row = 0 ("Smartphone")