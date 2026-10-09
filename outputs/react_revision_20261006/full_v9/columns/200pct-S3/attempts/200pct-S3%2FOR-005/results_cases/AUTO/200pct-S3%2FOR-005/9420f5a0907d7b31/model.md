ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: set of bread types, indexed by $i$ (from file_1_view_0, column item_name)

Parameters:
- $v_i$: expected profit per unit of bread type $i$ (file_1_view_0, column item_value)
- $a_i$: storage space required per unit of bread type $i$ (file_1_view_0, column resource_requirement)
- $C$: total available storage capacity (file_0_view_0, column resource_capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread type $i$ to order each day

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

- $I$: All records in file_1_view_0 (products.csv), column item_name
- $v_i$: file_1_view_0 (products.csv), column item_value, keyed by item_name
- $a_i$: file_1_view_0 (products.csv), column resource_requirement, keyed by item_name
- $C$: file_0_view_0 (capacity.csv), column resource_capacity
- $x_i$: Decision variable for each $i \in I$