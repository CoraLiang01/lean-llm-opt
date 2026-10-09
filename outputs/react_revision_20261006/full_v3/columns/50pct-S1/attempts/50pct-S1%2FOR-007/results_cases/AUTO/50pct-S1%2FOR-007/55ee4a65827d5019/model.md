##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), corresponding to all ProductName values in products.csv.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, nonnegative integer).
- $v_i$ = profit per unit of vehicle type $i$ (from Value column).
- $w_i$ = inventory weight per unit of vehicle type $i$ (from Weight column).
- $C$ = overall inventory capacity (from Capacity column in capacity.csv).

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

##### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: Decision variable for each $i \in I$.