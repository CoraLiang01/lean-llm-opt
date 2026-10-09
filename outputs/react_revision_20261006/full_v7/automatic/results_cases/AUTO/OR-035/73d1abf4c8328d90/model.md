#### Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from all ProductName in file_1_view_0)
- For each $i \in I$:
    - $v_i$ = expected profit per unit of bread $i$ (Value from file_1_view_0)
    - $w_i$ = storage weight per unit of bread $i$ (Weight from file_1_view_0)
- $C$ = total storage capacity (Capacity from file_0_view_0)
- $x_i$ = number of units of bread $i$ to order each day (decision variable, integer, $\geq 0$)

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName in file_1_view_0
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (bread type)