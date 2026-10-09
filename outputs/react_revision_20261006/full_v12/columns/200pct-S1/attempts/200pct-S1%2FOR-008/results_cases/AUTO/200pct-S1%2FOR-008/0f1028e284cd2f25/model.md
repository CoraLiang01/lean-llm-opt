#### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: benefit coefficient for vehicle type $i$ (from file_1_view_0, column Value, key ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (from file_0_view_0, column Capacity, key VehicleType)
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
   x_i \leq u_i \quad \forall i \in I
   \]
2. Total inventory capacity:
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. Integrality and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv, column VehicleType)
- $b_i$: file_1_view_0 (products.csv), column Value, key ProductName = $i$
- $u_i$: file_0_view_0 (capacity.csv), column Capacity, key VehicleType = $i$
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values in file_0_view_0)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)