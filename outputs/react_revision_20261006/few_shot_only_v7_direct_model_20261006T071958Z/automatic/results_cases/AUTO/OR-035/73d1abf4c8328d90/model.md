ABSTRACT MATHEMATICAL MODEL

Sets:
- $I$: set of bread types (from products.csv, column ProductName)

Parameters:
- $v_i$: expected profit per unit of bread $i$ (from products.csv, column Value)
- $w_i$: storage weight per unit of bread $i$ (from products.csv, column Weight)
- $C$: total storage capacity (from capacity.csv, column Capacity)

Decision Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: number of units of bread $i$ to order each day

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

- $I$: All records in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity (single record)
- $x_i$: decision variable for each $i \in I$