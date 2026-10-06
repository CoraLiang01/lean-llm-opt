#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (with business identifier VehicleID from capacity.csv and ProductName from products.csv).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Maximum daily inventory limit for vehicle type $i$.
- $C$: Total inventory capacity per day.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-type Inventory Limit:**
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All records in both capacity.csv (VehicleID, VehicleType) and products.csv (ProductName). Match vehicle types by VehicleType = ProductName.
- $b_i$: file_1_view_0.Value, where file_1_view_0.ProductName = file_0_view_0.VehicleType and file_0_view_0.VehicleID = $i$.
- $u_i$: file_0_view_0.Capacity, for VehicleID $i$.
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity over all $i$).
- $x_i$: Number of vehicles of type $i$ to order per day.

**Source Tables:**
- file_0_view_0: capacity.csv [VehicleID, VehicleType, Capacity]
- file_1_view_0: products.csv [ProductName, Value, Weight]

**Note:** All vehicle types and their limits/benefits are included as per the original data. The model uses the explicit business identifier VehicleID for all indexing and mapping.