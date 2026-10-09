Mathematical Model

Index Sets:
- Let $R$ be the set of shelves, indexed by $r$, with business identifier resource_id from file_0_view_0.
- Let $I$ be the set of products, indexed by $i$, with business identifier item_name from file_1_view_0.

Parameters:
- $v_i$: value of one unit of product $i$ (item_value from file_1_view_0)
- $a_i$: weight (resource requirement) of one unit of product $i$ (resource_requirement from file_1_view_0)
- $c_r$: capacity of shelf $r$ (resource_capacity from file_0_view_0)

Decision Variables:
- $x_{ri}$: number of units of product $i$ to place on shelf $r$, $x_{ri} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{r \in R} \sum_{i \in I} v_i x_{ri}
\]

Subject to:
\[
\sum_{i \in I} a_i x_{ri} \leq c_r \qquad \forall r \in R
\]
\[
x_{ri} \in \mathbb{Z}_{\geq 0} \qquad \forall r \in R,\, i \in I
\]

Data Mapping

- $R$: resource_id from file_0_view_0 (capacity.csv)
- $I$: item_name from file_1_view_0 (products.csv)
- $v_i$: item_value from file_1_view_0 (products.csv)
- $a_i$: resource_requirement from file_1_view_0 (products.csv)
- $c_r$: resource_capacity from file_0_view_0 (capacity.csv)
- $x_{ri}$: number of units of item_name $i$ on resource_id $r$ (decision variable, not in data)

All index sets, parameters, and constraints are mapped directly to the original file columns as specified.