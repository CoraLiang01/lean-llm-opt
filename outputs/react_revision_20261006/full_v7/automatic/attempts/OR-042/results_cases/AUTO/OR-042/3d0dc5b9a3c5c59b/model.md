##### Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$ (from all ProductName in products.csv)
- For each $i \in I$:
    - $v_i$ = Value of drug type $i$ (from Value in products.csv)
    - $w_i$ = Weight of drug type $i$ (from Weight in products.csv)
- $C$ = overall inventory capacity (from Capacity in capacity.csv)
- $x_i$ = number of units of drug type $i$ to order daily (decision variable, integer, $\geq 0$)

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

- $I$: All ProductName in table_id file_1_view_0 (products.csv)
- $v_i$: Value column in table_id file_1_view_0, keyed by ProductName
- $w_i$: Weight column in table_id file_1_view_0, keyed by ProductName
- $C$: Capacity column in table_id file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)