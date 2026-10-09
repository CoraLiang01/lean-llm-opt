#### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), with item names from file_1_view_0.item_name
- $x_i$ = number of units of bread type $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$
- $v_i$ = expected profit per unit of bread type $i$ (file_1_view_0.item_value)
- $a_i$ = storage requirement per unit of bread type $i$ (file_1_view_0.resource_requirement)
- $C$ = total storage capacity (file_0_view_0.resource_capacity)

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

#### Data Mapping

- $I$: All item_name in file_1_view_0
- $v_i$: file_1_view_0.item_value for item $i$
- $a_i$: file_1_view_0.resource_requirement for item $i$
- $C$: file_0_view_0.resource_capacity
- $x_i$: Number of units of bread type $i$ to order each day (decision variable, integer, nonnegative)