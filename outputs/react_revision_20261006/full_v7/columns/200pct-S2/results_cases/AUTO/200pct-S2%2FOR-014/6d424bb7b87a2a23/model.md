##### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$ = set of products, indexed by $p$ (from all ProductName in file_1_view_0)
- $x_{sp}$ = number of units of product $p$ placed on shelf $s$ (decision variable, integer, $\geq 0$)
- $v_p$ = value of product $p$ (from Value in file_1_view_0)
- $w_p$ = weight of product $p$ (from Weight in file_1_view_0)
- $C_s$ = capacity of shelf $s$ (from Capacity in file_0_view_0)

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

##### Data Mapping

- $S$: All ShelfID in file_0_view_0 (capacity.csv), column "ShelfID"
- $P$: All ProductName in file_1_view_0 (products.csv), column "ProductName"
- $v_p$: file_1_view_0, column "Value", keyed by "ProductName"
- $w_p$: file_1_view_0, column "Weight", keyed by "ProductName"
- $C_s$: file_0_view_0, column "Capacity", keyed by "ShelfID"
- $x_{sp}$: Decision variable for each $(s,p)$ pair

All indices, parameters, and constraints are mapped directly to the original data columns and business identifiers.