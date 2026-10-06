#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from products.csv, column ProductName)
- For each $i \in I$:
    - $v_i$ = Value of product $i$ (from products.csv, column Value)
    - $w_i$ = Weight of product $i$ (from products.csv, column Weight)
- $C$ = overall stock capacity (from capacity.csv, column Capacity)
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

- $I$ (products): file_1_view_0, column ProductName
- $v_i$: file_1_view_0, column Value, keyed by ProductName
- $w_i$: file_1_view_0, column Weight, keyed by ProductName
- $C$: file_0_view_0, column Capacity, row 0

All parameters and indices are mapped directly from the original CSV files and columns as returned.