## Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from products.csv)
- $u_i$ = daily inventory limit for vehicle type $i$ (from capacity.csv)

### Objective
\[
\max \sum_{i \in I} b_i x_i
\]

### Constraints

1. **Vehicle Type Inventory Limits**
   \[
   0 \leq x_i \leq u_i \qquad \forall i \in I
   \]
   (No more than the daily inventory limit for each vehicle type.)

2. **Integrality**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

## Data Mapping

- $I$: All vehicle types where VehicleType in file_0_view_0 and ProductName in file_1_view_0
- $b_i$: file_1_view_0, column "Value", with key ProductName $=i$
- $u_i$: file_0_view_0, column "Capacity", with key VehicleType $=i$
- $x_i$: Decision variable, integer, for each $i \in I$

No total inventory capacity constraint is present in the data; only per-vehicle-type limits are enforced.