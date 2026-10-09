Mathematical Model

Index Sets:
- Let $S$ be the set of shelves, with each shelf identified by resource_id from file_0_view_0.
- Let $P$ be the set of products, with each product identified by item_name from file_1_view_0.

Parameters:
- $c_s$: capacity of shelf $s$ (resource_capacity from file_0_view_0, indexed by resource_id $s \in S$)
- $v_p$: value of product $p$ (item_value from file_1_view_0, indexed by item_name $p \in P$)
- $w_p$: weight (resource requirement) of product $p$ (resource_requirement from file_1_view_0, indexed by item_name $p \in P$)

Decision Variables:
- $x_{sp}$: number of units of product $p$ to place on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$ for all $s \in S$, $p \in P$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \, x_{sp}
\]

Subject to:
- Shelf capacity constraints:
\[
\sum_{p \in P} w_p \, x_{sp} \leq c_s \qquad \forall s \in S
\]
- Integrality and nonnegativity:
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

Data Mapping

- $S$: All resource_id in file_0_view_0 (capacity.csv)
- $P$: All item_name in file_1_view_0 (products.csv)
- $c_s$: resource_capacity in file_0_view_0, indexed by resource_id
- $v_p$: item_value in file_1_view_0, indexed by item_name
- $w_p$: resource_requirement in file_1_view_0, indexed by item_name
- $x_{sp}$: number of units of product item_name $p$ on shelf resource_id $s$ (decision variable, not in data)

All parameters and index sets are defined directly from the current CSV data as described above.