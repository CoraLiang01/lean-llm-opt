#### Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from file_1_view_0, column ProductName)
- For each $i \in I$:
    - $p_i$ = expected profit per unit of bread $i$ (file_1_view_0, column Value)
    - $w_i$ = weight (storage space required) per unit of bread $i$ (file_1_view_0, column Weight)
- $C$ = total storage capacity (file_0_view_0, column Capacity)
- Decision variables: $x_i$ = number of units of bread $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} p_i x_i
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
- $p_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (bread type)