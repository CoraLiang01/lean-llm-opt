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

- $I$: All records where file_id = `file_0_view_0` column `VehicleType` and file_id = `file_1_view_0` column `ProductName` (matched by value).
- $b_i$: file_id = `file_1_view_0`, column `Value`, with $i$ matched to `ProductName`.
- $u_i$: file_id = `file_0_view_0`, column `Capacity`, with $i$ matched to `VehicleType`.
- $C$: $\sum_{i \in I} u_i$ (sum of all `Capacity` values in file_id = `file_0_view_0`).

---

**Notes:**
- Each $x_i$ is the number of vehicles of type $i$ to order per day (integer, nonnegative).
- The model maximizes total benefit, subject to per-type and total inventory limits.
- All index sets and parameters are defined directly from the supplied data; no values are omitted or invented.