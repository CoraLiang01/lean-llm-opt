### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: benefit coefficient for vehicle type $i$ (from file_1_view_0, column Value, with key ProductName = VehicleType)
- $u_i$: daily inventory limit for vehicle type $i$ (from file_0_view_0, column Capacity, with key VehicleType)
- $C$: total inventory capacity per day $= \sum_{i \in I} u_i$

**Decision Variables:**
- $x_i$: number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. Vehicle-type daily inventory limits:
   \[
   0 \leq x_i \leq u_i \quad \forall i \in I
   \]
2. Total inventory capacity:
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv, column VehicleType)
- $b_i$: file_1_view_0 (products.csv), column Value, with key ProductName = VehicleType
- $u_i$: file_0_view_0 (capacity.csv), column Capacity, with key VehicleType
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0, column Capacity)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)

All parameters are mapped directly from the returned CSV data using the exact column and table identifiers.