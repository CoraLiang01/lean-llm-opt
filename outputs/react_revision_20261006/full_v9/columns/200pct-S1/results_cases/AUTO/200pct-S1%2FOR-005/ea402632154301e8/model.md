#### Mathematical Model

Let $I$ be the set of bread types (indexed by $i$), with each $i$ corresponding to a unique item_name from file_1_view_0.

Let $x_i$ = number of units of bread type $i$ to order each day (decision variable, integer, $x_i \geq 0$).

Parameters:
- $v_i$ = expected profit per unit of bread type $i$ (item_value, file_1_view_0)
- $a_i$ = storage space required per unit of bread type $i$ (resource_requirement, file_1_view_0)
- $C$ = total storage capacity (resource_capacity, file_0_view_0)

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

- $I$: All item_name in file_1_view_0 (products.csv)
- $v_i$: item_value in file_1_view_0, mapped by item_name
- $a_i$: resource_requirement in file_1_view_0, mapped by item_name
- $C$: resource_capacity in file_0_view_0 (capacity.csv)
- $x_i$: integer decision variable for each item_name in file_1_view_0

All parameters and index sets are defined directly from the returned CSV data.