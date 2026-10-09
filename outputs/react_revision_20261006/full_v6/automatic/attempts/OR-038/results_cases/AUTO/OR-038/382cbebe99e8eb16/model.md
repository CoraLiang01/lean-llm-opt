#### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$ (from file_1_view_0, column Value, with key ProductName = VehicleType)
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$ (from file_0_view_0, column Capacity, with key VehicleType)
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-vehicle-type daily inventory limits:**
   \[
   0 \leq x_i \leq u_i, \quad \forall i \in I
   \]

2. **Total inventory capacity constraint:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
   (If the "total inventory capacity" is meant to be the sum of all per-type capacities; if a separate total capacity is provided, replace the right-hand side accordingly.)

3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: Set of vehicle types from file_0_view_0 (capacity.csv), column VehicleType
- $u_i$: file_0_view_0, column Capacity, key VehicleType
- $b_i$: file_1_view_0, column Value, key ProductName = VehicleType
- $x_i$: Decision variable for each $i \in I$

All parameters are mapped using the exact column names and table_ids as above.