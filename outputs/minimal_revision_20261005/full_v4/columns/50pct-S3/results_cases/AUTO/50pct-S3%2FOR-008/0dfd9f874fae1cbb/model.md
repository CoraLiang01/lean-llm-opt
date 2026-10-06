**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in both tables).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

---

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-vehicle-type daily inventory limits:**
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. **Total daily inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All vehicle types where VehicleType in `file_0_view_0` and ProductName in `file_1_view_0`.
- $b_i$: `file_1_view_0`, column `Value`, with key `ProductName = i`.
- $u_i$: `file_0_view_0`, column `Capacity`, with key `VehicleType = i`.

---

**Notes:**
- The total daily inventory capacity is the sum of all per-type capacities, as no separate global limit is provided.
- Each $x_i$ is an integer variable representing the number of vehicles of type $i$ to order per day.