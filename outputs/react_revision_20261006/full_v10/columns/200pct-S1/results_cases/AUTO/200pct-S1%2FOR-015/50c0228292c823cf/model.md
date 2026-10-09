Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, with each shelf identified by resource_id from file_0_view_0.
- Let $P$ be the set of products, with each product identified by item_name from file_1_view_0.

Parameters:
- $v_p$: value of product $p \in P$ (item_value from file_1_view_0)
- $w_p$: weight (space requirement) of product $p \in P$ (resource_requirement from file_1_view_0)
- $C_s$: capacity of shelf $s \in S$ (resource_capacity from file_0_view_0)

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Constraints:
1. Shelf capacity constraints (for each shelf $s \in S$):
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]

2. Nonnegativity and integrality:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: All resource_id in file_0_view_0 (capacity.csv)
- $P$: All item_name in file_1_view_0 (products.csv)
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $w_p$: file_1_view_0.resource_requirement, keyed by item_name
- $C_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $x_{sp}$: Number of units of product item_name $p$ on shelf resource_id $s$ (decision variable, integer, $\geq 0$)