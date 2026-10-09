Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, indexed by $s$, with shelf identifiers resource_id from file_0_view_0.
- Let $P$ be the set of products, indexed by $p$, with product identifiers item_name from file_1_view_0.

Parameters:
- $v_p$: value of product $p$ (item_value from file_1_view_0, indexed by item_name)
- $w_p$: weight (resource requirement) of product $p$ (resource_requirement from file_1_view_0, indexed by item_name)
- $C_s$: capacity of shelf $s$ (resource_capacity from file_0_view_0, indexed by resource_id)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \, x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: resource_id from file_0_view_0 (capacity.csv)
- $P$: item_name from file_1_view_0 (products.csv)
- $v_p$: item_value from file_1_view_0, indexed by item_name
- $w_p$: resource_requirement from file_1_view_0, indexed by item_name
- $C_s$: resource_capacity from file_0_view_0, indexed by resource_id
- $x_{s,p}$: number of units of product $p$ placed on shelf $s$ (decision variable, nonnegative integer)