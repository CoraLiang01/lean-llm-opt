ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of vehicle types (indexed by $i$), from file_1_view_0 ProductName.

Parameters:
- $v_i$: benefit (Value) of vehicle type $i$, from file_1_view_0 Value.
- $w_i$: weight (Weight) of vehicle type $i$, from file_1_view_0 Weight.
- $C$: total inventory capacity, from file_0_view_0 Capacity.

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of vehicle type $i$ to order daily.

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

DATA MAPPING

- $I$: All records in file_1_view_0, column ProductName.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $C$: file_0_view_0, column Capacity (single value).
- $x_i$: decision variable for each $i \in I$.