#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$
- $x_i$ = number of units of product $i$ to order each day (decision variable, $x_i \in \mathbb{Z}_{\geq 0}$)
- $v_i$ = value (benefit) per unit of product $i$
- $w_i$ = weight (stock space required) per unit of product $i$
- $C$ = overall stock capacity

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

- $I$: All products in file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, for each $i$
- $w_i$: file_1_view_0, column Weight, for each $i$
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable, number of units of product $i$ to order each day

Each $x_i$ is a nonnegative integer. The model maximizes total benefit subject to the overall stock capacity.