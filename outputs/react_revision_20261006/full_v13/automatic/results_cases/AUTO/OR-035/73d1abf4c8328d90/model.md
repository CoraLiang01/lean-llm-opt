#### Mathematical Model

Let:
- $I$ = set of bread types, indexed by $i$ (from ProductName in products.csv)
- $x_i$ = number of units of bread type $i$ to order each day (integer, $\geq 0$)
- $v_i$ = expected profit per unit of bread type $i$ (from Value in products.csv)
- $w_i$ = storage weight per unit of bread type $i$ (from Weight in products.csv)
- $C$ = total storage capacity (from Capacity in capacity.csv)

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

- $I$: All ProductName values in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, mapped by ProductName
- $w_i$: Weight column in file_1_view_0, mapped by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0

- Decision variables $x_i$ are indexed by ProductName from file_1_view_0

All parameters and index sets are defined directly from the current CSV data as described above.