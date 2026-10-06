#### Abstract Mathematical Model

Let:
- $I$ = set of drug products, indexed by $i$ (with identifiers ProductName from products.csv)
- For each $i \in I$:
    - $v_i$ = Value of product $i$ (from products.csv, column Value)
    - $w_i$ = Weight of product $i$ (from products.csv, column Weight)
- $C$ = overall stock capacity (from capacity.csv, column Capacity)
- Decision variables: $x_i$ = number of units of drug $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$ (drug products): file_1_view_0, column ProductName
- $v_i$ (benefit per unit): file_1_view_0, column Value, keyed by ProductName
- $w_i$ (weight per unit): file_1_view_0, column Weight, keyed by ProductName
- $C$ (overall stock capacity): file_0_view_0, column Capacity
- $x_i$ (decision variable): number of units of product $i$ to order each day, indexed by ProductName

All parameters and sets are defined directly from the returned rows and columns.