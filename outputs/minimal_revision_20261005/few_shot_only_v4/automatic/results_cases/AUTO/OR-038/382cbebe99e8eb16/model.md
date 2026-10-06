**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from capacity.csv and products.csv, matched by VehicleType/ProductName).

**Parameters:**
- $b_i$: Benefit coefficient (Value) for vehicle type $i$ (from products.csv, column Value, matched by ProductName = VehicleType).
- $u_i$: Maximum daily inventory limit for vehicle type $i$ (from capacity.csv, column Capacity, indexed by VehicleID/VehicleType).
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable).

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in I$

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Daily Inventory Limit for Each Vehicle Type:**
   \[
   x_i \leq u_i \qquad \forall i \in I
   \]
2. **Total Inventory Capacity Constraint:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
   (If the "total inventory capacity" is meant to be the sum of all individual capacities; if a separate total capacity is provided, use that value instead.)

3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$ (vehicle types):  
  - capacity.csv: VehicleID, VehicleType  
  - products.csv: ProductName

- $b_i$ (benefit coefficient):  
  - products.csv: Value (matched by ProductName = VehicleType)

- $u_i$ (daily inventory limit):  
  - capacity.csv: Capacity (matched by VehicleType)

- $x_i$ (decision variable):  
  - Number of vehicles of type $i$ to order per day

**Source Columns Used:**
- capacity.csv: VehicleID, VehicleType, Capacity
- products.csv: ProductName, Value

**Notes:**
- All vehicle types in capacity.csv and products.csv are matched by VehicleType/ProductName.
- Each $x_i$ is a nonnegative integer, as required.
- The model maximizes total benefit subject to per-type and total inventory limits.