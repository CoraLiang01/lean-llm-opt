ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $S$: set of shelves, indexed by $s$ (from file_0_view_0, column resource_id)
- $P$: set of products, indexed by $p$ (from file_1_view_0, column item_name)

Parameters:
- $c_s$: capacity of shelf $s$ (from file_0_view_0, column resource_capacity)
- $v_p$: value per unit of product $p$ (from file_1_view_0, column item_value)
- $w_p$: weight (space requirement) per unit of product $p$ (from file_1_view_0, column resource_requirement)

Decision Variables:
- $x_{sp} \in \mathbb{Z}_{\geq 0}$: number of units of product $p$ placed on shelf $s$

Objective:
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

Subject to:
\[
\sum_{p \in P} w_p \cdot x_{sp} \leq c_s \qquad \forall s \in S
\]
\[
x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
\]

DATA MAPPING

- $S$ (shelves): file_0_view_0, column resource_id
- $P$ (products): file_1_view_0, column item_name
- $c_s$: file_0_view_0, column resource_capacity, keyed by resource_id
- $v_p$: file_1_view_0, column item_value, keyed by item_name
- $w_p$: file_1_view_0, column resource_requirement, keyed by item_name
- $x_{sp}$: number of units of product $p$ on shelf $s$ (decision variable, integer, nonnegative)