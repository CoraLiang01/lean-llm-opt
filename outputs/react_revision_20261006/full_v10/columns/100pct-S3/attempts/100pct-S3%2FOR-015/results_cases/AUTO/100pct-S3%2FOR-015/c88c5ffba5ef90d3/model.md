ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $R$ be the set of shelves, indexed by $r$, with business identifier resource_id from file_0_view_0.
- Let $I$ be the set of products, indexed by $i$, with business identifier item_name from file_1_view_0.

Parameters:
- $v_i$: value per unit of product $i$ (item_value, file_1_view_0)
- $a_i$: weight (resource requirement) per unit of product $i$ (resource_requirement, file_1_view_0)
- $c_r$: capacity of shelf $r$ (resource_capacity, file_0_view_0)

Decision Variables:
- $x_{ri} \in \mathbb{Z}_{\geq 0}$: number of units of product $i$ allocated to shelf $r$

Objective:
\[
\max \sum_{r \in R} \sum_{i \in I} v_i \, x_{ri}
\]

Subject to:
\[
\sum_{i \in I} a_i \, x_{ri} \leq c_r \qquad \forall r \in R
\]
\[
x_{ri} \in \mathbb{Z}_{\geq 0} \qquad \forall r \in R,\, i \in I
\]

DATA MAPPING

- $R$: resource_id from file_0_view_0 (capacity.csv)
- $I$: item_name from file_1_view_0 (products.csv)
- $v_i$: item_value from file_1_view_0, indexed by item_name
- $a_i$: resource_requirement from file_1_view_0, indexed by item_name
- $c_r$: resource_capacity from file_0_view_0, indexed by resource_id
- $x_{ri}$: number of units of item_name $i$ on shelf resource_id $r$ (decision variable, integer, $\geq 0$)