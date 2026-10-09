Mathematical Model

Index Sets:
- $S$: set of shelves (indexed by $s$), with shelf IDs from file_0_view_0.resource_id
- $P$: set of products (indexed by $p$), with product IDs from file_1_view_0.item_name

Parameters:
- $C_s$: capacity of shelf $s$ (file_0_view_0.resource_capacity)
- $v_p$: value of product $p$ (file_1_view_0.item_value)
- $w_p$: weight (space requirement) of product $p$ (file_1_view_0.resource_requirement)

Decision Variables:
- $x_{sp}$: number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

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

Data Mapping

- $S$: file_0_view_0.resource_id
- $P$: file_1_view_0.item_name
- $C_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $w_p$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{sp}$: decision variable for allocation of product $p$ to shelf $s$