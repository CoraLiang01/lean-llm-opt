**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType in file_0_view_0 and ProductName in file_1_view_0).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$.

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-Vehicle-Type Inventory Limits:**
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

- $I$: All VehicleType in `file_0_view_0` (capacity.csv, column: VehicleType) and all ProductName in `file_1_view_0` (products.csv, column: ProductName). Match by name.
- $u_i$: `file_0_view_0`, column: Capacity, for VehicleType $i$.
- $b_i$: `file_1_view_0`, column: Value, for ProductName $i$.

**Note:** The mapping between vehicle types and product names is by exact string match between VehicleType (capacity.csv) and ProductName (products.csv). All vehicle types present in both files are included in $I$.