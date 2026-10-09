##### Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$ (from ProductName in file_1_view_0)
- $x_i$ = number of units of drug type $i$ to order daily (integer, $\geq 0$)
- $v_i$ = benefit coefficient of drug type $i$ (Value in file_1_view_0)
- $w_i$ = weight per unit of drug type $i$ (Weight in file_1_view_0)
- $C$ = overall inventory capacity (Capacity in file_0_view_0)

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

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)