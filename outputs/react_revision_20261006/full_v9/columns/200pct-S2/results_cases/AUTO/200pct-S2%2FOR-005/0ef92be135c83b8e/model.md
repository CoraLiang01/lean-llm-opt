#### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), corresponding to all item_name in file_1_view_0.
- $x_i$ = number of units of bread type $i$ to order each day; $x_i \in \mathbb{Z}_{\geq 0}$.

Parameters:
- $v_i$ = expected profit per unit of bread type $i$ (item_value from file_1_view_0).
- $a_i$ = storage requirement per unit of bread type $i$ (resource_requirement from file_1_view_0).
- $C$ = total storage capacity (resource_capacity from file_0_view_0).

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

---

#### Data Mapping

- $I$: All item_name in file_1_view_0.
- $v_i$: item_value in file_1_view_0, mapped by item_name.
- $a_i$: resource_requirement in file_1_view_0, mapped by item_name.
- $C$: resource_capacity in file_0_view_0.
- $x_i$: Decision variable for each item_name in file_1_view_0.