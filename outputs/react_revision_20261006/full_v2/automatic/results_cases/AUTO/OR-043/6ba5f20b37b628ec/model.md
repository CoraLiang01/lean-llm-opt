##### Mathematical Model

Let:
- $I$ = set of drug products, indexed by $i$ (from all ProductName in file_1_view_0)
- $x_i$ = number of units of drug $i$ to order each day (decision variable, nonnegative integer)
- $v_i$ = benefit per unit of drug $i$ (from Value in file_1_view_0)
- $w_i$ = weight (stock space requirement) per unit of drug $i$ (from Weight in file_1_view_0)
- $C$ = overall stock capacity (from Capacity in file_0_view_0)

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

- $I$: All ProductName in file_1_view_0
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)