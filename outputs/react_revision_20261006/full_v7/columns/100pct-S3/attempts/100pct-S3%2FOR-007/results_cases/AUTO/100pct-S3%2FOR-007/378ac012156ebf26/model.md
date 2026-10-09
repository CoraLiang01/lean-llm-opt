#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from file_1_view_0[ProductName]
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$, from file_1_view_0[Value]
    - $w_i$ = inventory weight per unit of vehicle $i$, from file_1_view_0[Weight]
- $C$ = total inventory capacity, from file_0_view_0[Capacity]
- Decision variable: $x_i$ = number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} p_i x_i
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

- $I$ (vehicle types): file_1_view_0[ProductName]
- $p_i$ (profit per unit): file_1_view_0[Value]
- $w_i$ (weight per unit): file_1_view_0[Weight]
- $C$ (total capacity): file_0_view_0[Capacity]
- Decision variable $x_i$: number of vehicles of type $i$ to order per day

All parameters are mapped directly from the corresponding columns in the returned tables.