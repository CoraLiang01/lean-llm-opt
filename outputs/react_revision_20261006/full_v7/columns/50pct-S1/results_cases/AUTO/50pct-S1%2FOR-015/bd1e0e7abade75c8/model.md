Mathematical Model

Index Sets:
- $R$: set of shelves (resource_id in file_0_view_0)
- $J$: set of products (item_name in file_1_view_0)

Parameters:
- $c_r$: capacity of shelf $r$ (resource_capacity in file_0_view_0)
- $v_j$: value of product $j$ (item_value in file_1_view_0)
- $a_j$: weight (resource requirement) of product $j$ (resource_requirement in file_1_view_0)

Decision Variables:
- $x_{rj}$: number of units of product $j$ placed on shelf $r$, $x_{rj} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{r \in R} \sum_{j \in J} v_j x_{rj}
\]

Subject to:
\[
\sum_{j \in J} a_j x_{rj} \leq c_r \qquad \forall r \in R
\]
\[
x_{rj} \in \mathbb{Z}_{\geq 0} \qquad \forall r \in R,\, j \in J
\]

Data Mapping

- $R$: file_0_view_0.resource_id
- $J$: file_1_view_0.item_name
- $c_r$: file_0_view_0.resource_capacity, keyed by resource_id
- $v_j$: file_1_view_0.item_value, keyed by item_name
- $a_j$: file_1_view_0.resource_requirement, keyed by item_name
- $x_{rj}$: number of units of product $j$ placed on shelf $r$ (decision variable, indexed by resource_id and item_name)