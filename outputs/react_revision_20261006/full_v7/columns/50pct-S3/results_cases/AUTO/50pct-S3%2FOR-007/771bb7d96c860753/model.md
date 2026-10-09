#### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), as given by the ProductName column in products.csv.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value column, products.csv)
    - $w_i$ = inventory weight per unit of vehicle $i$ (Weight column, products.csv)
- $C$ = overall inventory capacity (Capacity column, capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

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

- $I$: All ProductName values from file_1_view_0 (products.csv, column ProductName)
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)