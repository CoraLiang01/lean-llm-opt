#### Abstract Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from file_1_view_0, column item_name)
- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- $v_i$ = expected profit per unit of bread type $i$ (from file_1_view_0, column item_value)
- $a_i$ = storage space required per unit of bread type $i$ (from file_1_view_0, column resource_requirement)
- $C$ = total available storage capacity (from file_0_view_0, column resource_capacity)

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

- $I$ (bread types): file_1_view_0, column item_name
- $v_i$ (expected profit): file_1_view_0, column item_value, keyed by item_name
- $a_i$ (storage requirement): file_1_view_0, column resource_requirement, keyed by item_name
- $C$ (storage capacity): file_0_view_0, column resource_capacity

Each $x_i$ is the integer number of units of bread type $i$ to order each day. The model maximizes total expected profit, subject to the bakery's storage capacity.