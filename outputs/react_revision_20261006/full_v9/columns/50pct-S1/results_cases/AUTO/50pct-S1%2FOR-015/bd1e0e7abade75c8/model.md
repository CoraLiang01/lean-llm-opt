ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves (indexed by $s$), with business identifier resource_id from file_0_view_0 (capacity.csv)
- $P$: set of products (indexed by $p$), with business identifier item_name from file_1_view_0 (products.csv)

Parameters:
- $v_p$: value of one unit of product $p$ (item_value from file_1_view_0)
- $w_p$: resource requirement (weight) of one unit of product $p$ (resource_requirement from file_1_view_0)
- $C_s$: capacity of shelf $s$ (resource_capacity from file_0_view_0)

Decision Variables:
- $x_{s,p} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ allocated to shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{s,p}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{s,p} \leq C_s \qquad \forall s \in S
\]
\[
x_{s,p} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$: All resource_id in file_0_view_0 (capacity.csv)
- $P$: All item_name in file_1_view_0 (products.csv)
- $v_p$: item_value from file_1_view_0, keyed by item_name
- $w_p$: resource_requirement from file_1_view_0, keyed by item_name
- $C_s$: resource_capacity from file_0_view_0, keyed by resource_id
- $x_{s,p}$: number of units of product $p$ (item_name) to allocate to shelf $s$ (resource_id)

All variables $x_{s,p}$ are nonnegative integers. Every shelf's total allocated product weight cannot exceed its resource_capacity. The objective is to maximize total value across all shelves and products.