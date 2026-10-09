Mathematical Model

Index Sets:
- $S$: set of displays (ShelfID from file_0_view_0)
- $P$: set of products (ProductName from file_1_view_0)

Parameters:
- $c_s$: capacity of display $s$ (Capacity from file_0_view_0, indexed by ShelfID)
- $v_p$: value of product $p$ (Value from file_1_view_0, indexed by ProductName)
- $w_p$: weight of product $p$ (Weight from file_1_view_0, indexed by ProductName)

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on display $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
1. Display capacity constraints (for all $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s
\]
2. Minimum allocation of the first product (let $p^*$ be the ProductName in the first row of file_1_view_0):
\[
\sum_{s \in S} x_{s p^*} \geq 5
\]
3. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \quad \forall s \in S,\, p \in P
\]

Data Mapping

Index Sets:
- $S$: ShelfID from file_0_view_0 (capacity.csv)
- $P$: ProductName from file_1_view_0 (products.csv)

Parameters:
- $c_s$: file_0_view_0, column Capacity, key ShelfID
- $v_p$: file_1_view_0, column Value, key ProductName
- $w_p$: file_1_view_0, column Weight, key ProductName

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on display $s$ (nonnegative integer)

Special constraint:
- $p^*$ is the ProductName in file_1_view_0, source_row 0 (the first product listed).