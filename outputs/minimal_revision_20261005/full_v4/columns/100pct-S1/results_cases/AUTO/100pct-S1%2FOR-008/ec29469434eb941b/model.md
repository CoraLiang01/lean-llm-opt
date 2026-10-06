**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0 and ProductName in file_1_view_0).

**Parameters:**
- $b_i$: Benefit coefficient of vehicle type $i$ (from Value in file_1_view_0, matched by $i = $ ProductName).
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$ (from Capacity in file_0_view_0, matched by $i = $ VehicleType).
- $C$: Total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$.

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

---

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**

1. **Vehicle Type Inventory Limits:**
   \[
   x_i \leq u_i \qquad \forall i \in I
   \]

2. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]

3. **Integrality and Nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All VehicleType in `file_0_view_0` (capacity.csv) and all ProductName in `file_1_view_0` (products.csv).
- $b_i$: `file_1_view_0`, column `Value`, with $i$ matched to `ProductName`.
- $u_i$: `file_0_view_0`, column `Capacity`, with $i$ matched to `VehicleType`.
- $C$: $\sum_{i \in I} u_i$ (sum of all `Capacity` in `file_0_view_0`).
- $x_i$: Decision variable for each $i \in I$.

**Note:** All parameters are mapped directly from the returned CSV rows using the exact column and table identifiers. No data is omitted or synthesized.