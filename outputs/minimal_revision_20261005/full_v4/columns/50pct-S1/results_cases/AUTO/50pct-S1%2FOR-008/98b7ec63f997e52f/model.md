**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0 and ProductName in file_1_view_0).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$ (from Value in file_1_view_0, matched by ProductName = VehicleType).
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$ (from Capacity in file_0_view_0, indexed by VehicleType).
- $U$: Total inventory capacity per day, defined as $U = \sum_{i \in I} u_i$.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

---

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**
1. **Total Inventory Capacity Constraint:**
   \[
   \sum_{i \in I} x_i \leq U
   \]
2. **Vehicle Type Inventory Limits:**
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
3. **Integrality and Nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All VehicleType in `file_0_view_0` (capacity.csv, column VehicleType) and all ProductName in `file_1_view_0` (products.csv, column ProductName).
- $b_i$: `file_1_view_0`, column Value, matched by ProductName = VehicleType.
- $u_i$: `file_0_view_0`, column Capacity, indexed by VehicleType.
- $U$: $\sum_{i \in I} u_i$ (sum of all Capacity in `file_0_view_0`).
- $x_i$: Decision variable for each $i \in I$.

**Note:** All index sets, parameters, and constraints are defined using the exact columns and identifiers from the returned CSV data. No data is omitted or synthesized.