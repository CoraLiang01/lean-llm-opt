Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), with each $i$ corresponding to a unique ProductName from file_1_view_0.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = profit per unit of vehicle type $i$ (parameter: Value from file_1_view_0).
- $w_i$ = weight (inventory space required) per unit of vehicle type $i$ (parameter: Weight from file_1_view_0).
- $C$ = overall inventory capacity (parameter: Capacity from file_0_view_0).

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

Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value
- $x_i$: Decision variable for each $i \in I$