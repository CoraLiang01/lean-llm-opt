Mathematical Model

Index Sets:
- Let I be the set of bread types, with each bread type identified by item_name from file_1_view_0.

Parameters:
- $v_i$: expected profit per unit of bread type $i$ (item_value from file_1_view_0)
- $a_i$: storage space required per unit of bread type $i$ (resource_requirement from file_1_view_0)
- $C$: total available storage capacity (resource_capacity from file_0_view_0)

Decision Variables:
- $x_i$: number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} a_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- $I$: All item_name in file_1_view_0
- $v_i$: item_value in file_1_view_0, indexed by item_name
- $a_i$: resource_requirement in file_1_view_0, indexed by item_name
- $C$: resource_capacity in file_0_view_0, row 0
- $x_i$: decision variable for each item_name in file_1_view_0

All parameters and index sets are mapped directly from the returned CSV data as specified.