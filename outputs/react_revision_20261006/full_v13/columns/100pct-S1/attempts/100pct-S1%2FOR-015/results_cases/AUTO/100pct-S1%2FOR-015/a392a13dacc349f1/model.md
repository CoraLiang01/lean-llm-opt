ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves (from file_0_view_0.resource_id)
- $P$: set of products (from file_1_view_0.item_name)

Parameters:
- $C_s$: capacity of shelf $s \in S$ (from file_0_view_0.resource_capacity)
- $v_p$: value of product $p \in P$ (from file_1_view_0.item_value)
- $a_p$: weight (resource requirement) of product $p \in P$ (from file_1_view_0.resource_requirement)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
\[
\sum_{p \in P} a_p \, x_{sp} \leq C_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$: All file_0_view_0.resource_id
- $P$: All file_1_view_0.item_name
- $C_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $a_p$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{sp}$: number of units of product $p$ on shelf $s$ (decision variable, integer, nonnegative)