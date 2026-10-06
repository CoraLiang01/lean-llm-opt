**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (maximum units that can be ordered per day) for vehicle type $i$.
- $C$: Total inventory capacity per day.

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

- $I$: All VehicleType values from `file_0_view_0` (capacity.csv, column "VehicleType")
- $b_i$: For each $i \in I$, $b_i$ is the Value from `file_1_view_0` (products.csv, column "Value") where ProductName = VehicleType
- $u_i$: For each $i \in I$, $u_i$ is the Capacity from `file_0_view_0` (capacity.csv, column "Capacity")
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values from `file_0_view_0`)

---

**Table References:**

- `file_0_view_0` (capacity.csv): columns ["VehicleType", "Capacity"]
- `file_1_view_0` (products.csv): columns ["ProductName", "Value"]

---

**Notes:**
- Each $x_i$ is an integer variable representing the number of vehicles of type $i$ to order per day.
- The model maximizes total benefit, subject to per-type and total inventory limits.