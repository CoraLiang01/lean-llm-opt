##### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$.
- $C$: Total inventory capacity per day.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-vehicle-type daily limit:**
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. **Total inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All VehicleType values from file_0_view_0.VehicleType.
- $b_i$: file_1_view_0.Value, matched where file_1_view_0.ProductName = file_0_view_0.VehicleType.
- $u_i$: file_0_view_0.Capacity.
- $C$: $\sum_{i \in I} u_i$ (the sum of all file_0_view_0.Capacity).
- $x_i$: Decision variable for each $i \in I$.

Each $x_i$ is the number of vehicles of type $i$ to order per day. The benefit coefficients and capacities are mapped by matching VehicleType/ProductName across the two files. The total inventory capacity $C$ is the sum of all per-type capacities.