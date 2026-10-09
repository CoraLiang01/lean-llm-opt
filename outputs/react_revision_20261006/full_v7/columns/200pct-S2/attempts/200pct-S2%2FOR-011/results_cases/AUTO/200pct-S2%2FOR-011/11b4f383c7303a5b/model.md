#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$, with identifiers ProductName from file_1_view_0.
- $x_i$ = number of units of product $i$ to order each day, $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = Value of product $i$ (from file_1_view_0, column Value).
- $w_i$ = Weight of product $i$ (from file_1_view_0, column Weight).
- $C$ = overall stock capacity (from file_0_view_0, column Capacity).

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

#### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity (capacity.csv)
- $x_i$: Decision variable, number of units to order for each $i \in I$