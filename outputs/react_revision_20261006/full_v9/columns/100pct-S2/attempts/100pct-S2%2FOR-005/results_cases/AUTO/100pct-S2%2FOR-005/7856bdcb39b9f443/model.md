#### Mathematical Model

Let:
- $I$ = set of bread types (indexed by $i$), from file_1_view_0[item_name]
- $x_i$ = number of units of bread type $i$ to order each day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$
- $v_i$ = expected profit per unit of bread type $i$, from file_1_view_0[item_value]
- $a_i$ = storage requirement per unit of bread type $i$, from file_1_view_0[resource_requirement]
- $C$ = total storage capacity, from file_0_view_0[resource_capacity]

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

- $I$: file_1_view_0[item_name]
- $v_i$: file_1_view_0[item_value], mapped by item_name
- $a_i$: file_1_view_0[resource_requirement], mapped by item_name
- $C$: file_0_view_0[resource_capacity]
- $x_i$: integer, for each $i \in I$