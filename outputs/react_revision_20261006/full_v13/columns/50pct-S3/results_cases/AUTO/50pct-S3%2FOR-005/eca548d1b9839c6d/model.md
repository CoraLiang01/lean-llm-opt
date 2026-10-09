#### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), from file_1_view_0.item_name
- For each $i \in I$:
    - $v_i$ = expected profit per unit of bread $i$ (file_1_view_0.item_value)
    - $a_i$ = storage space required per unit of bread $i$ (file_1_view_0.resource_requirement)
- $C$ = total storage capacity (file_0_view_0.resource_capacity)
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

- $I$ (bread types): file_1_view_0.item_name
- $v_i$ (expected profit): file_1_view_0.item_value
- $a_i$ (storage requirement): file_1_view_0.resource_requirement
- $C$ (storage capacity): file_0_view_0.resource_capacity
- $x_i$ (decision variable): number of units of bread $i$ to order each day (integer, $\geq 0$)