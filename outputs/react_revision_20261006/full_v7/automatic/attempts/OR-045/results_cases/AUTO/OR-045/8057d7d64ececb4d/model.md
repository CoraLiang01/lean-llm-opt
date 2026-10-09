#### Mathematical Model

Let:
- $I$ = set of produce types, indexed by $i$ (from all ProductName in file_1_view_0)
- $x_i$ = number of units of produce $i$ to order daily (integer, $x_i \geq 0$)
- $v_i$ = value (benefit) per unit of produce $i$ (from Value in file_1_view_0)
- $w_i$ = weight per unit of produce $i$ (from Weight in file_1_view_0)
- $C$ = overall inventory capacity (from Capacity in file_0_view_0)

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

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv)
- $x_i$: Decision variable for each $i \in I$ (integer, nonnegative)