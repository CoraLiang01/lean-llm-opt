Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, indexed by $s$, with shelf identifiers given by file_0_view_0.resource_id.
- Let $P$ be the set of products, indexed by $p$, with product identifiers given by file_1_view_0.item_name.

Parameters:
- $c_s$: capacity of shelf $s$ (file_0_view_0.resource_capacity)
- $v_p$: value of product $p$ (file_1_view_0.item_value)
- $a_p$: weight (space requirement) of product $p$ (file_1_view_0.resource_requirement)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ to place on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} a_p \, x_{s,p} \leq c_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: file_0_view_0.resource_id
- $P$: file_1_view_0.item_name
- $c_s$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_p$: file_1_view_0.item_value, keyed by item_name
- $a_p$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{s,p}$: number of units of product $p$ to place on shelf $s$ (decision variable, indexed by resource_id and item_name)