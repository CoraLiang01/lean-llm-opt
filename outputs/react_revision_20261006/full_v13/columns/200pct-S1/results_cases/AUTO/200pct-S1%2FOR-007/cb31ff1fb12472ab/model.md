#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from file_1_view_0[ProductName]
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (file_1_view_0[Value])
    - $w_i$ = weight (inventory space) per unit of vehicle $i$ (file_1_view_0[Weight])
- $C$ = total inventory capacity (file_0_view_0[Capacity])
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

#### Data Mapping

- $I$: All records in file_1_view_0[ProductName]
- $p_i$: file_1_view_0[Value], mapped by ProductName
- $w_i$: file_1_view_0[Weight], mapped by ProductName
- $C$: file_0_view_0[Capacity] (single value)
- $x_i$: Decision variable for each $i \in I$ (vehicle type from file_1_view_0[ProductName])