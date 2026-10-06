#### Abstract Mathematical Model

Let:
- $I$ = set of drug types, indexed by $i$ (from ProductName in products.csv)
- For each $i \in I$:
    - $v_i$ = benefit coefficient of drug type $i$ (from Value)
    - $w_i$ = weight per unit of drug type $i$ (from Weight)
- $C$ = overall inventory capacity (from Capacity in capacity.csv)
- Decision variables: $x_i$ = number of units of drug type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$ (drug types): file_1_view_0.ProductName
- $v_i$ (benefit coefficient): file_1_view_0.Value, keyed by ProductName
- $w_i$ (weight per unit): file_1_view_0.Weight, keyed by ProductName
- $C$ (overall capacity): file_0_view_0.Capacity

All parameters and index sets are mapped directly from the retrieved CSV data using the exact column and table identifiers above.