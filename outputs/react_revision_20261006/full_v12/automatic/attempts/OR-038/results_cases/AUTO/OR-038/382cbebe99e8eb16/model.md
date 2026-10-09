### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$, with business identifier $\text{VehicleID}$ from capacity.csv.

**Parameters:**
- $b_i$: benefit coefficient of vehicle type $i$ (from products.csv, column Value, matched by VehicleType = ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (from capacity.csv, column Capacity)
- $C$: total inventory capacity $= \sum_{i \in I} u_i$

**Decision Variables:**
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

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
3. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

### Data Mapping

- $I$: All $\text{VehicleID}$ in capacity.csv (file_0_view_0)
- $b_i$: products.csv (file_1_view_0), column Value, matched where capacity.csv VehicleType = products.csv ProductName
- $u_i$: capacity.csv (file_0_view_0), column Capacity
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity in capacity.csv)
- $x_i$: Decision variable for each $\text{VehicleID}$

All parameters are mapped using the exact column names and business identifiers from the source files.