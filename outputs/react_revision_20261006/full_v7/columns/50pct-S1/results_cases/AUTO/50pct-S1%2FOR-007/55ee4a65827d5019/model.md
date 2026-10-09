##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from the ProductName column in products.csv.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value column, products.csv)
    - $w_i$ = inventory space required per unit of vehicle $i$ (Weight column, products.csv)
- $C$ = total inventory capacity (Capacity column, capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

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

##### Data Mapping

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $p_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)