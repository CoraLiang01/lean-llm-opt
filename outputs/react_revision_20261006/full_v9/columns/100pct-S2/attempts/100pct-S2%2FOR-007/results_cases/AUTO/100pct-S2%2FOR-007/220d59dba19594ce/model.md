##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from file_1_view_0[ProductName]
- For each $i \in I$:
    - $v_i$ = profit per unit of vehicle $i$ (file_1_view_0[Value])
    - $w_i$ = inventory weight per unit of vehicle $i$ (file_1_view_0[Weight])
- $C$ = overall inventory capacity (file_0_view_0[Capacity])
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)

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

##### Data Mapping

- $I$: All records in file_1_view_0[ProductName]
- $v_i$: file_1_view_0[Value], matched by $i$
- $w_i$: file_1_view_0[Weight], matched by $i$
- $C$: file_0_view_0[Capacity]
- $x_i$: Decision variable for each $i \in I$ (vehicle type)