ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of drug types, indexed by $i$ (from all ProductName in products.csv)

Parameters:
- $v_i$: benefit coefficient of drug $i$ (Value column in products.csv)
- $w_i$: weight per unit of drug $i$ (Weight column in products.csv)
- $C$: total inventory capacity (Capacity column in capacity.csv)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of drug $i$ to order daily

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

- $I$: All records ProductName in table_id file_1_view_0, column ProductName
- $v_i$: table_id file_1_view_0, column Value, keyed by ProductName
- $w_i$: table_id file_1_view_0, column Weight, keyed by ProductName
- $C$: table_id file_0_view_0, column Capacity (single value)
- $x_i$: decision variable for each $i \in I$