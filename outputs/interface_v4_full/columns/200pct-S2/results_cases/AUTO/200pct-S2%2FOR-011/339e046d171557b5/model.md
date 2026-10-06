#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$, with ProductName from file_1_view_0 (products.csv)
- For each $i \in I$:
    - $v_i$ = Value of product $i$ (from Value column, file_1_view_0)
    - $w_i$ = Weight of product $i$ (from Weight column, file_1_view_0)
- $C$ = overall stock capacity (from Capacity column, file_0_view_0)
- Decision variables: $x_i$ = number of units of product $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} v_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (products): file_1_view_0, ProductName
- $v_i$: file_1_view_0, Value
- $w_i$: file_1_view_0, Weight
- $C$: file_0_view_0, Capacity

Each $x_i$ is the number of units of product $i$ to order each day. The total weight of all ordered products cannot exceed the overall stock capacity. The objective is to maximize total value.