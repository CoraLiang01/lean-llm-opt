**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0 and ProductName in file_1_view_0)

**Parameters:**
- $b_i$: benefit coefficient of vehicle type $i$ (from Value in file_1_view_0, matched by $i = $ ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (from Capacity in file_0_view_0, matched by $i = $ VehicleType)

**Decision Variables:**
- $x_i$: number of vehicles of type $i$ to order per day; $x_i \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**

1. **Vehicle Type Daily Inventory Limits:**
   \[
   x_i \leq u_i \qquad \forall i \in I
   \]

2. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]

3. **Integrality and Nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All VehicleType in `file_0_view_0` and ProductName in `file_1_view_0`
- $b_i$: `file_1_view_0`, column `Value`, with $i$ = `ProductName`
- $u_i$: `file_0_view_0`, column `Capacity`, with $i$ = `VehicleType`