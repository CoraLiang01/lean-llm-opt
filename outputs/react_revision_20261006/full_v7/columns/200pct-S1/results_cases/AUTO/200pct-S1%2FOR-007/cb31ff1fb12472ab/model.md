##### Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), corresponding to all ProductName values in products.csv.
- For each $i \in I$:
    - $p_i$ = profit per unit of vehicle $i$ (Value from products.csv)
    - $w_i$ = weight (inventory space required) per unit of vehicle $i$ (Weight from products.csv)
- $C$ = overall inventory capacity (Capacity from capacity.csv)
- Decision variables: $x_i$ = number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to:
$$
\sum_{i \in I} w_i x_i \leq C
$$

$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

---

##### Data Mapping

- $I$: All ProductName values from file_1_view_0 (products.csv, column ProductName)
- $p_i$: file_1_view_0, column Value, for each $i$
- $w_i$: file_1_view_0, column Weight, for each $i$
- $C$: file_0_view_0, column Capacity
- $x_i$: Decision variable for each $i \in I$ (vehicle type)