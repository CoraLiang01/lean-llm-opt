#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$, with business identifier VehicleID from file_0_view_0.
- $b_i$ = benefit coefficient for vehicle type $i$ (from file_1_view_0, matched by VehicleType/ProductName).
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$ (from file_0_view_0, column Capacity).
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day.
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable).

$\text{Maximize} \quad \sum_{i \in I} b_i x_i$

$\text{subject to:}$

$\quad \sum_{i \in I} x_i \leq C$

$\quad 0 \leq x_i \leq u_i \qquad \forall i \in I$

$\quad x_i \in \mathbb{Z} \qquad \forall i \in I$

---

#### Data Mapping

- Index set $I$: All VehicleID from file_0_view_0 (capacity.csv), column VehicleID.
- Parameter $b_i$: file_1_view_0 (products.csv), column Value, matched where file_0_view_0.VehicleType = file_1_view_0.ProductName.
- Parameter $u_i$: file_0_view_0 (capacity.csv), column Capacity.
- Total capacity $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity).
- Decision variable $x_i$: number of vehicles of type $i$ to order per day, for each $i \in I$.

All variables and parameters are indexed by the business identifier VehicleID from file_0_view_0.