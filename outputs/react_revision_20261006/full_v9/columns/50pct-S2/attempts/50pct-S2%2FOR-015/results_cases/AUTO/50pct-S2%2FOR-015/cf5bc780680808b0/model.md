ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves (indexed by $s$), from file_0_view_0.resource_id
- $P$: set of products (indexed by $p$), from file_1_view_0.item_name

Parameters:
- $C_s$: capacity of shelf $s$, from file_0_view_0.resource_capacity
- $v_p$: value of product $p$, from file_1_view_0.item_value
- $w_p$: weight (space requirement) of product $p$, from file_1_view_0.resource_requirement

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

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

DATA MAPPING

- $S$: file_0_view_0.resource_id
- $P$: file_1_view_0.item_name
- $C_s$: file_0_view_0.resource_capacity (for shelf $s$)
- $v_p$: file_1_view_0.item_value (for product $p$)
- $w_p$: file_1_view_0.resource_requirement (for product $p$)
- $x_{sp}$: number of units of product $p$ to place on shelf $s$ (decision variable, integer, $\geq 0$)