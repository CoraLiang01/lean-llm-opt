## Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from products.csv, column ProductName)
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (products.csv, column Value)
    - $w_i$ = inventory weight per unit of vehicle $i$ (products.csv, column Weight)
- $C$ = overall inventory capacity (capacity.csv, column Capacity)
- Decision variables: $x_i$ = number of vehicles of type $i$ to order per day

### Objective
$$
\max \sum_{i \in I} p_i x_i
$$

### Constraints
1. **Overall Inventory Capacity:**
$$
\sum_{i \in I} w_i x_i \leq C
$$

2. **Nonnegativity and Integrality:**
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

## Data Mapping

- $I$ (vehicle types): file_1_view_0, column ProductName
- $p_i$ (profit per unit): file_1_view_0, column Value
- $w_i$ (weight per unit): file_1_view_0, column Weight
- $C$ (overall capacity): file_0_view_0, column Capacity

All data is used in source order, with original identifiers and columns.