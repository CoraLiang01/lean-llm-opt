Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, with each shelf identified by resource_id from file_0_view_0.
- Let $P$ be the set of products, with each product identified by item_name from file_1_view_0.

Parameters:
- $v_p$: value of product $p \in P$ (item_value from file_1_view_0)
- $w_p$: weight (resource requirement) of product $p \in P$ (resource_requirement from file_1_view_0)
- $C_s$: capacity of shelf $s \in S$ (resource_capacity from file_0_view_0)

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

- $S$: All resource_id in file_0_view_0 (capacity.csv)
- $P$: All item_name in file_1_view_0 (products.csv)
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name
- $C_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $x_{s,p}$: Decision variable for each $(s,p) \in S \times P$ (integer, nonnegative)