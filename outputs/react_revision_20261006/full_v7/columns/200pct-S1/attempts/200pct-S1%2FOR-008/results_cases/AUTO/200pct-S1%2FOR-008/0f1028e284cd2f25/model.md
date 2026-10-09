##### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

$\text{Maximize} \quad \sum_{i \in I} b_i x_i$

$\text{subject to:}$

$\sum_{i \in I} x_i \leq C$

$0 \leq x_i \leq u_i \qquad \forall i \in I$

$x_i \in \mathbb{Z} \qquad \forall i \in I$

---

##### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv) and all ProductName values from file_1_view_0 (products.csv)
- $b_i$: Value column from file_1_view_0 (products.csv), matched to vehicle type $i$ by ProductName = VehicleType
- $u_i$: Capacity column from file_0_view_0 (capacity.csv), matched to vehicle type $i$ by VehicleType
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values from file_0_view_0)
- $x_i$: integer variable for each $i \in I$ (vehicle type)

All indices, parameters, and constraints are mapped directly to the columns and rows of the provided CSV files.