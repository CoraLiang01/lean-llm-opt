#### Abstract Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$ (with business identifier ProductName from products.csv).

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from Value column, products.csv)
- $w_i$: inventory weight (space requirement) for vehicle type $i$ (from Weight column, products.csv)
- $C$: total inventory capacity (from Capacity column, capacity.csv)

Decision variables:
- $x_i$: number of units of vehicle type $i$ to order daily ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All ProductName values in file_1_view_0 (products.csv), in source order.
- $b_i$: file_1_view_0, column Value, indexed by ProductName.
- $w_i$: file_1_view_0, column Weight, indexed by ProductName.
- $C$: file_0_view_0, column Capacity.

All parameters and index sets are mapped directly from the CSVQA_DATA bindings above, preserving original file and row order.