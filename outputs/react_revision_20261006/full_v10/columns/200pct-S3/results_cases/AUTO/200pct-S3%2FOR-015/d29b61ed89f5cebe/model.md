Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, with each shelf identified by resource_id from file_0_view_0.
- Let $P$ be the set of products, with each product identified by item_name from file_1_view_0.

Parameters:
- $v_p$: value of product $p \in P$ (item_value from file_1_view_0)
- $w_p$: weight (resource requirement) of product $p \in P$ (resource_requirement from file_1_view_0)
- $C_s$: capacity of shelf $s \in S$ (resource_capacity from file_0_view_0)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

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

- $S$: file_0_view_0.resource_id
- $P$: file_1_view_0.item_name
- $v_p$: file_1_view_0.item_value (for product $p$)
- $w_p$: file_1_view_0.resource_requirement (for product $p$)
- $C_s$: file_0_view_0.resource_capacity (for shelf $s$)
- $x_{sp}$: number of units of product $p$ on shelf $s$ (decision variable, indexed by $s$ and $p$)