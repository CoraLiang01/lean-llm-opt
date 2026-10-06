#### Abstract Mathematical Model

Let:
- $I$ = set of vehicle types (indexed by $i$), with business identifier ProductName from products.csv.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable), $x_i \in \mathbb{Z}_{\geq 0}$.

Parameters:
- $p_i$ = profit per unit of vehicle $i$ (from Value column, products.csv).
- $w_i$ = weight (inventory space requirement) per unit of vehicle $i$ (from Weight column, products.csv).
- $C$ = total inventory capacity (from Capacity column, capacity.csv).

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

#### Data Mapping

- $p_i$: file_1_view_0, ProductName, Value
- $w_i$: file_1_view_0, ProductName, Weight
- $C$: file_0_view_0, Capacity

Index set $I$ is all ProductName values in file_1_view_0 (products.csv).