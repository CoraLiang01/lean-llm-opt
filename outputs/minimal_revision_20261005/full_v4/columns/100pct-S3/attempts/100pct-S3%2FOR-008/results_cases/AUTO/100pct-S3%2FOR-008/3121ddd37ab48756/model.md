**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$.
- $C$: Total inventory capacity per day (sum of all $u_i$).

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Vehicle Type Inventory Limits:**
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]
3. **Integrality and Nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All VehicleType in `file_0_view_0` (capacity.csv) and all ProductName in `file_1_view_0` (products.csv). Matched by VehicleType = ProductName.
- $b_i$: `Value` column from `file_1_view_0` (products.csv), matched by ProductName.
- $u_i$: `Capacity` column from `file_0_view_0` (capacity.csv), matched by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all `Capacity` in `file_0_view_0`).

**Table References:**
- `file_0_view_0` (capacity.csv): columns [VehicleType, Capacity]
- `file_1_view_0` (products.csv): columns [ProductName, Value]