#### Abstract Mathematical Model

Let:
- $I$ = set of areas (indexed by $i$), corresponding to all ProductName values in products.csv.
- $x_i$ = integer variable: scale of development in area $i$ per day.

Parameters:
- $b_i$ = benefit coefficient for area $i$ (from products.csv, Value column).
- $w_i$ = development unit weight for area $i$ (from products.csv, Weight column).
- $C$ = overall development capacity (from capacity.csv, Capacity column).

Objective:
$$
\max \sum_{i \in I} b_i x_i
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

- $I$ (areas): file_1_view_0.ProductName
- $b_i$: file_1_view_0.Value (for each ProductName $i$)
- $w_i$: file_1_view_0.Weight (for each ProductName $i$)
- $C$: file_0_view_0.Capacity

- Decision variables $x_i$ are indexed by file_1_view_0.ProductName and represent the integer scale of development in each area per day.