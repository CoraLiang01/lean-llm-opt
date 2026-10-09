ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column ShelfID)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column ProductName)

Parameters:
- $v_p$: value per unit of product $p$ (from file_1_view_0, column Value)
- $w_p$: weight per unit of product $p$ (from file_1_view_0, column Weight)
- $C_s$: capacity (maximum total weight) of shelf $s$ (from file_0_view_0, column Capacity)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
1. Shelf capacity constraints (for each shelf $s$):
\[
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
\]
2. Nonnegativity and integrality:
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$ (shelves): file_0_view_0, column ShelfID
- $P$ (products): file_1_view_0, column ProductName
- $v_p$: file_1_view_0, column Value, keyed by ProductName
- $w_p$: file_1_view_0, column Weight, keyed by ProductName
- $C_s$: file_0_view_0, column Capacity, keyed by ShelfID
- $x_{s,p}$: allocation variable for shelf $s$ and product $p$ (decision variable, not in data)

All index sets, parameters, and constraints are defined directly from the supplied data. No data is omitted or synthesized.