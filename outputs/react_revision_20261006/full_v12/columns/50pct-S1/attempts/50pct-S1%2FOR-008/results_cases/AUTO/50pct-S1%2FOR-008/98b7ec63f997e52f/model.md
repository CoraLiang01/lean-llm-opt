## Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

**Decision variables:**
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

**Parameters:**
- $b_i$: benefit coefficient for vehicle type $i$ (from file_1_view_0, column Value)
- $u_i$: daily inventory limit for vehicle type $i$ (from file_0_view_0, column Capacity)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. Vehicle-type-specific daily inventory limits:
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. Total inventory capacity:
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
3. Integrality and nonnegativity:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

## Data Mapping

- $I$: All records in file_0_view_0 (capacity.csv), column VehicleID (with VehicleType as label)
- $b_i$: file_1_view_0 (products.csv), column Value, matched to $i$ by VehicleType = ProductName
- $u_i$: file_0_view_0 (capacity.csv), column Capacity, indexed by VehicleID
- $x_i$: decision variable for VehicleID $i$ (VehicleType as label)

- Constraint 1: $x_i \leq u_i$ for each VehicleID in file_0_view_0
- Constraint 2: $\sum_{i} x_i \leq \sum_{i} u_i$ (sum over all VehicleIDs in file_0_view_0)
- Objective: $\sum_{i} b_i x_i$ (sum over all VehicleIDs in file_0_view_0, with $b_i$ from file_1_view_0 matched by VehicleType = ProductName)