## Abstract Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from products.csv ProductName.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.
- $v_i$ = profit per vehicle of type $i$ (Value from products.csv).
- $w_i$ = inventory weight per vehicle of type $i$ (Weight from products.csv).
- $C$ = overall inventory capacity (Capacity from capacity.csv).

### Objective
$$
\max \sum_{i \in I} v_i x_i
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

### Data Mapping

- $I$ (vehicle types): file_1_view_0, column ProductName
- $v_i$ (profit): file_1_view_0, column Value, keyed by ProductName
- $w_i$ (weight): file_1_view_0, column Weight, keyed by ProductName
- $C$ (capacity): file_0_view_0, column Capacity

All data is used in source order and with original identifiers.