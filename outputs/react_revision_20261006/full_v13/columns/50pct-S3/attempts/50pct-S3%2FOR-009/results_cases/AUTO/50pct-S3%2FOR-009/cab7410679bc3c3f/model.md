ABSTRACT MATHEMATICAL MODEL

Let:
- $I$ = set of areas, indexed by $i$, with area names given by the ProductName column in file_1_view_0.
- For each area $i \in I$:
    - $v_i$ = Value of developing area $i$ (from Value column in file_1_view_0)
    - $w_i$ = Resource required to develop area $i$ (from Weight column in file_1_view_0)
- $C$ = overall development capacity (from Capacity column in file_0_view_0)
- Decision variable: $x_i$ = scale of development per day in area $i$ (continuous, $x_i \geq 0$; integer if the scale must be in whole units)

Objective:
\[
\max \sum_{i \in I} v_i x_i
\]

Subject to:
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

Data Mapping:
- $I$: file_1_view_0.ProductName
- $v_i$: file_1_view_0.Value, keyed by ProductName
- $w_i$: file_1_view_0.Weight, keyed by ProductName
- $C$: file_0_view_0.Capacity
- $x_i$: scale of development per day in area $i$ (decision variable, indexed by file_1_view_0.ProductName)