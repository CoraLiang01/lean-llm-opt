**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0 and ProductName in file_1_view_0).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
  - Data Mapping: $b_i \leftarrow$ Value where ProductName $= i$ in file_1_view_0
- $u_i$: Maximum daily inventory limit for vehicle type $i$.
  - Data Mapping: $u_i \leftarrow$ Capacity where VehicleType $= i$ in file_0_view_0

**Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-vehicle-type daily inventory limits:**
   \[
   x_i \leq u_i \qquad \forall i \in I
   \]
2. **Total inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All VehicleType in file_0_view_0 and ProductName in file_1_view_0 (matched by name).
- $b_i$: file_1_view_0, column Value, key ProductName.
- $u_i$: file_0_view_0, column Capacity, key VehicleType.

---

**Notes:**
- All vehicle types present in both files are included.
- Each $x_i$ is a nonnegative integer, as required.
- The total inventory capacity is the sum of all per-type capacities, as no separate total is provided.