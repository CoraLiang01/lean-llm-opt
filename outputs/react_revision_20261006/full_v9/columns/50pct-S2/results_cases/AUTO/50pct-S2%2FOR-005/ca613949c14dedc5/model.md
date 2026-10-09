#### Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from all item_name in file_1_view_0)
- For each $i \in I$:
    - $v_i$ = expected profit per unit of bread $i$ (item_value from file_1_view_0)
    - $a_i$ = storage requirement per unit of bread $i$ (resource_requirement from file_1_view_0)
- $C$ = total storage capacity (resource_capacity from file_0_view_0)
- $x_i$ = number of units of bread $i$ to order each day (decision variable, integer, $x_i \geq 0$)

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

- $I$: All item_name in file_1_view_0
- $v_i$: item_value (file_1_view_0, column "item_value", key "item_name")
- $a_i$: resource_requirement (file_1_view_0, column "resource_requirement", key "item_name")
- $C$: resource_capacity (file_0_view_0, column "resource_capacity")
- $x_i$: integer decision variable for each $i \in I$