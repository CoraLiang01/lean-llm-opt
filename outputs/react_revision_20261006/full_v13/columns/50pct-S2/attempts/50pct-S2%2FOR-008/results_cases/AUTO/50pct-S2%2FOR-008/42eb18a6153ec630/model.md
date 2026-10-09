### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0)
- $b_i$ = benefit coefficient for vehicle type $i$ (from Value in file_1_view_0, matched by VehicleType/ProductName)
- $u_i$ = daily inventory limit for vehicle type $i$ (from Capacity in file_0_view_0)
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day (sum of all per-type capacities)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)

#### Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

#### Constraints:
1. **Per-vehicle-type daily inventory limits:**
   \[
   0 \leq x_i \leq u_i \quad \forall i \in I
   \]
2. **Total inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

### Data Mapping

- $I$: All VehicleType in file_0_view_0 (capacity.csv), preserving source order.
- $b_i$: Value from file_1_view_0 (products.csv), matched where ProductName = VehicleType.
- $u_i$: Capacity from file_0_view_0 (capacity.csv), for each VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity in file_0_view_0).
- $x_i$: Decision variable for each VehicleType in $I$.

**Table references:**
- file_0_view_0: capacity.csv, columns [VehicleType, Capacity]
- file_1_view_0: products.csv, columns [ProductName, Value]

All indices, parameters, and constraints are mapped directly from the returned data, with no omitted vehicle types or constraints.