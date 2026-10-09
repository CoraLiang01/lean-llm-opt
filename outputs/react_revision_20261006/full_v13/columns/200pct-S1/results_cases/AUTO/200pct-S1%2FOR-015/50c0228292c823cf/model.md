ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

Parameters:
- $v_p$: value per unit of product $p$ (from file_1_view_0, column item_value)
- $w_p$: weight (space requirement) per unit of product $p$ (from file_1_view_0, column resource_requirement)
- $C_s$: capacity (maximum total weight) of shelf $s$ (from file_0_view_0, column resource_capacity)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to place on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$: file_0_view_0, column resource_id
- $P$: file_1_view_0, column item_name
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name
- $C_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $x_{sp}$: number of units of product $p$ on shelf $s$ (decision variable, integer, nonnegative)