#### Mathematical Model

Let:
- $S$ = set of shelves, indexed by $s$ (from all ShelfID in file_0_view_0)
- $P$ = set of products, indexed by $p$ (from all ProductName in file_1_view_0)

Parameters:
- $v_p$ = value of one unit of product $p$ (Value from file_1_view_0)
- $w_p$ = weight of one unit of product $p$ (Weight from file_1_view_0)
- $C_s$ = capacity of shelf $s$ (Capacity from file_0_view_0)

Decision variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

---

#### Data Mapping

- $S$ (shelves): All ShelfID in file_0_view_0 (capacity.csv)
- $P$ (products): All ProductName in file_1_view_0 (products.csv)
- $v_p$: file_1_view_0, column "Value", keyed by ProductName
- $w_p$: file_1_view_0, column "Weight", keyed by ProductName
- $C_s$: file_0_view_0, column "Capacity", keyed by ShelfID
- $x_{s,p}$: integer, nonnegative, for all $(s,p) \in S \times P$