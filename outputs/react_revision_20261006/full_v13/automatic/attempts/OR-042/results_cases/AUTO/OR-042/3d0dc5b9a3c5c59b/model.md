##### Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$ (from all ProductName in file_1_view_0)
- For each $i \in I$:
    - $v_i$ = Value of drug $i$ (from Value in file_1_view_0)
    - $w_i$ = Weight per unit of drug $i$ (from Weight in file_1_view_0)
- $C$ = overall inventory capacity (from Capacity in file_0_view_0)
- $x_i$ = number of units of drug $i$ to order daily (decision variable, integer, $x_i \geq 0$)

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

- $I$: All ProductName in file_1_view_0 (products.csv, column ProductName)
- $v_i$: file_1_view_0, column Value, for each $i$
- $w_i$: file_1_view_0, column Weight, for each $i$
- $C$: file_0_view_0, column Capacity (capacity.csv)
- $x_i$: integer variable for each $i \in I$