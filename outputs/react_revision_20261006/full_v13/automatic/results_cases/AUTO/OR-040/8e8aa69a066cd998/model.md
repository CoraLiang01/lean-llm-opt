ABSTRACT MATHEMATICAL MODEL

Let:
- $I$ = set of areas, indexed by $i$ (from all ProductName in file_1_view_0)
- $b_i$ = benefit coefficient for area $i$ (Value from file_1_view_0, column Value)
- $w_i$ = resource usage per unit for area $i$ (Weight from file_1_view_0, column Weight)
- $C$ = overall development capacity (Capacity from file_0_view_0, column Capacity)
- $x_i$ = integer variable: daily scale of development in area $i$

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

DATA MAPPING

- $I$: All ProductName in file_1_view_0 (products.csv, column ProductName)
- $b_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv, row 0)
- $x_i$: integer, nonnegative, for each $i \in I$