##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), as given by all ProductName in products.csv.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value from products.csv)
    - $w_i$ = inventory weight per unit of vehicle $i$ (Weight from products.csv)
- $C$ = overall inventory capacity (Capacity from capacity.csv)
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

- $I$: All ProductName in file_1_view_0 (products.csv), preserve source order.
- $p_i$: Value column in file_1_view_0, keyed by ProductName.
- $w_i$: Weight column in file_1_view_0, keyed by ProductName.
- $C$: Capacity column in file_0_view_0 (capacity.csv), single value.
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are mapped directly from the returned CSV data, with no omitted entities or constraints.