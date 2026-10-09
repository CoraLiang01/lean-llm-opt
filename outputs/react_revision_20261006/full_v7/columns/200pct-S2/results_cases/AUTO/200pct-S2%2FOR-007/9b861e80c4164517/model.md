#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), with each $i$ corresponding to a unique ProductName from file_1_view_0.
- For each $i \in I$:
  - $p_i$ = profit per unit of vehicle $i$ (Value from file_1_view_0)
  - $w_i$ = inventory space required per unit of vehicle $i$ (Weight from file_1_view_0)
- $C$ = overall inventory capacity (Capacity from file_0_view_0)
- Decision variables: $x_i$ = number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $p_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (vehicle type)