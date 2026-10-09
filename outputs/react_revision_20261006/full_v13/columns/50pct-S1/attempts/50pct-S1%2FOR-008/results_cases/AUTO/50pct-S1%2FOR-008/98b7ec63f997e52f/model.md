## Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (with identifiers from file_0_view_0.VehicleType and file_1_view_0.ProductName)
- $b_i$ = benefit coefficient for vehicle type $i$ (from file_1_view_0.Value)
- $u_i$ = daily inventory limit for vehicle type $i$ (from file_0_view_0.Capacity)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)

Define total inventory capacity as $C = \sum_{i \in I} u_i$.

### Objective
\[
\max \sum_{i \in I} b_i x_i
\]

### Constraints

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

## Data Mapping

- $I$: All vehicle types from file_0_view_0.VehicleType and file_1_view_0.ProductName.
- $b_i$: file_1_view_0.Value, matched to $i$ by ProductName = VehicleType.
- $u_i$: file_0_view_0.Capacity, matched to $i$ by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity).
- $x_i$: Decision variable for each $i \in I$.

All parameters are mapped directly from the returned CSV data using the exact column and table identifiers.