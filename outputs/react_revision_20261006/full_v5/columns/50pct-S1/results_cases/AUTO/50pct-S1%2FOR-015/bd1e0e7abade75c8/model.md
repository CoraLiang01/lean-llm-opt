Mathematical Model

Index Sets:
- $S$: set of shelves (indexed by $s$), with shelf IDs given by file_0_view_0.resource_id
- $P$: set of products (indexed by $p$), with product IDs given by file_1_view_0.item_name

Parameters:
- $C_s$: capacity of shelf $s$ (file_0_view_0.resource_capacity, indexed by resource_id)
- $v_p$: value of product $p$ (file_1_view_0.item_value, indexed by item_name)
- $w_p$: weight (resource requirement) of product $p$ (file_1_view_0.resource_requirement, indexed by item_name)

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

Data Mapping

- $S$: file_0_view_0.resource_id
- $P$: file_1_view_0.item_name
- $C_s$: file_0_view_0.resource_capacity (indexed by resource_id)
- $v_p$: file_1_view_0.item_value (indexed by item_name)
- $w_p$: file_1_view_0.resource_requirement (indexed by item_name)
- $x_{sp}$: number of units of product $p$ placed on shelf $s$ (decision variable, not in data)