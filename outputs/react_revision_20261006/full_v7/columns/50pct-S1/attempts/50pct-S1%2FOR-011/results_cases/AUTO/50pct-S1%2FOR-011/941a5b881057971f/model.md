#### Mathematical Model

Let:
- $I$ = set of products (indexed by $i$), with each product identified by its ProductName.
- $x_i$ = number of units of product $i$ to order each day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = Value of product $i$ (from products.csv).
- $w_i$ = Weight of product $i$ (from products.csv).
- $C$ = overall stock capacity (from capacity.csv).

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

- $I$: All products in file_1_view_0, column ProductName.
- $v_i$: file_1_view_0, column Value, for each $i$.
- $w_i$: file_1_view_0, column Weight, for each $i$.
- $C$: file_0_view_0, column Capacity (single value).
- $x_i$: Decision variable for each $i \in I$ (nonnegative integer).