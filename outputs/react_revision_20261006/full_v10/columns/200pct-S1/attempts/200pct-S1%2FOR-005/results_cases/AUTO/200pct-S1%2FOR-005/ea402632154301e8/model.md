ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $I$ be the set of bread types, with each $i \in I$ corresponding to item_name from file_1_view_0.

Parameters:
- $v_i$: expected profit per unit of bread type $i$ (item_value from file_1_view_0)
- $a_i$: storage space required per unit of bread type $i$ (resource_requirement from file_1_view_0)
- $C$: total available storage capacity (resource_capacity from file_0_view_0)

Decision Variables:
- $x_i$: number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$

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

Index Sets:
- $I$: All item_name in file_1_view_0

Parameters:
- $v_i$: file_1_view_0.item_value for bread type $i$
- $a_i$: file_1_view_0.resource_requirement for bread type $i$
- $C$: file_0_view_0.resource_capacity

Decision Variables:
- $x_i$: number of units to order of bread type $i$ (indexed by file_1_view_0.item_name)