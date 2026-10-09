ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of produce types (indexed by $i$), from all ProductName in products.csv.

Parameters:
- $w_i$: weight per unit of produce $i$ (from Weight in products.csv).
- $v_i$: value per unit of produce $i$ (from Value in products.csv).
- $C$: total inventory capacity (from Capacity in capacity.csv).

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of produce $i$ to order daily.

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

- $I$: All ProductName in table_id file_1_view_0, column ProductName.
- $w_i$: file_1_view_0, column Weight, keyed by ProductName.
- $v_i$: file_1_view_0, column Value, keyed by ProductName.
- $C$: file_0_view_0, column Capacity.
- $x_i$: decision variable for each $i \in I$.