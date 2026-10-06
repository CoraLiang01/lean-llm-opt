**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (VehicleType from file_0_view_0 and ProductName from file_1_view_0).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$ (Value from file_1_view_0, ProductName = VehicleType).
- $u_i$: Daily inventory limit for vehicle type $i$ (Capacity from file_0_view_0, VehicleType).
- $C$: Total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Vehicle Type Daily Inventory Limits:**
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

- $I$: All VehicleType in file_0_view_0 and all ProductName in file_1_view_0 (matched by name).
- $b_i$: file_1_view_0, column Value, with $i$ = ProductName.
- $u_i$: file_0_view_0, column Capacity, with $i$ = VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0, column Capacity).
- $x_i$: Decision variable for each $i \in I$.

**Table References:**
- file_0_view_0: ["VehicleType", "Capacity"]
- file_1_view_0: ["ProductName", "Value"]

**Notes:**
- Each $x_i$ is an integer variable representing the number of vehicles of type $i$ to order per day.
- The model uses all vehicle types present in both files, matched by name.