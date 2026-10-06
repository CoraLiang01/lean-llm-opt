## Abstract Mathematical Model

Let:
- $I$ = set of drug products, indexed by $i$ (from products.csv, column ProductName)
- For each $i \in I$:
    - $v_i$ = benefit per unit of drug $i$ (products.csv, column Value)
    - $w_i$ = weight (stock space required) per unit of drug $i$ (products.csv, column Weight)
- $C$ = overall stock capacity (capacity.csv, column Capacity)
- Decision variables: $x_i$ = number of units of drug $i$ to order each day ($x_i \in \mathbb{Z}_{\geq 0}$)

### Objective
$$
\max \sum_{i \in I} v_i x_i
$$

### Constraints
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

## Data Mapping

- $I$ (drug products): file_1_view_0, column ProductName
- $v_i$ (benefit): file_1_view_0, column Value, key ProductName
- $w_i$ (weight): file_1_view_0, column Weight, key ProductName
- $C$ (overall stock capacity): file_0_view_0, column Capacity

All parameters and index sets are to be taken directly from the referenced columns and rows, preserving their original order and identifiers.