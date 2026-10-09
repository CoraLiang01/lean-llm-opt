#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all ProductName in products.csv)
- For each $i \in I$:
    - $v_i$ = Value of product $i$ (from Value in products.csv)
    - $w_i$ = Weight of product $i$ (from Weight in products.csv)
- $C$ = overall stock capacity (from Capacity in capacity.csv)
- $x_i$ = number of units of product $i$ to order each day (decision variable, nonnegative integer)

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

---

#### Data Mapping

- $I$: All records in file_1_view_0 (products.csv), column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity (scalar)
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer)