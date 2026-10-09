ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of bread types, with elements $i$ corresponding to each item_name in file_1_view_0.

Parameters:
- $v_i$: expected profit per unit of bread type $i$ (item_value, file_1_view_0, column item_value, key item_name).
- $a_i$: storage space required per unit of bread type $i$ (resource_requirement, file_1_view_0, column resource_requirement, key item_name).
- $C$: total available storage capacity (resource_capacity, file_0_view_0, column resource_capacity).

Decision Variables:
- $x_i$: number of units of bread type $i$ to order each day; $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$.

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} a_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

DATA MAPPING

- $I$: All item_name in file_1_view_0.
- $v_i$: file_1_view_0, column item_value, key item_name.
- $a_i$: file_1_view_0, column resource_requirement, key item_name.
- $C$: file_0_view_0, column resource_capacity.
- $x_i$: Decision variable for each $i \in I$.