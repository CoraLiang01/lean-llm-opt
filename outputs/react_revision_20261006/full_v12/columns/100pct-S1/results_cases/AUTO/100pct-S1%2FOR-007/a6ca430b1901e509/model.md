### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), from products.csv ProductName.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value, products.csv)
    - $w_i$ = inventory space required per unit of vehicle $i$ (Weight, products.csv)
- $C$ = overall inventory capacity (Capacity, capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

#### Decision Variables
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

#### Objective
Maximize total profit:
$$
\max \sum_{i \in I} p_i x_i
$$

#### Constraints
Overall inventory capacity:
$$
\sum_{i \in I} w_i x_i \leq C
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

---

### Data Mapping

- $I$: All ProductName in file_1_view_0 (products.csv)
- $p_i$: Value column in file_1_view_0, keyed by ProductName
- $w_i$: Weight column in file_1_view_0, keyed by ProductName
- $C$: Capacity column in file_0_view_0 (capacity.csv), row 0
- $x_i$: Decision variable for each $i \in I$ (vehicle type)

All parameters are mapped directly from the returned CSV data, preserving original identifiers.