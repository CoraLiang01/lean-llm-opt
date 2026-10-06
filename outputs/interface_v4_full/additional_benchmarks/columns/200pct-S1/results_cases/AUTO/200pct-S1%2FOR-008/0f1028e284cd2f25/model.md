#### Abstract Mathematical Model

**Sets:**
- $I$: Set of vehicle types, indexed by $i$.

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$.  
  (from products.csv, column Value, table_id: file_1_view_0, key: ProductName)
- $u_i$: Maximum daily inventory limit for vehicle type $i$.  
  (from capacity.csv, column Capacity, table_id: file_0_view_0, key: VehicleType)
- $U$: Total inventory capacity per day (user-supplied or policy parameter).

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day.  
  ($x_i \in \mathbb{Z}_{\geq 0}$, integer and nonnegative)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-Vehicle-Type Inventory Limits:**  
   $\quad x_i \leq u_i \quad \forall i \in I$

2. **Total Inventory Capacity:**  
   $\quad \sum_{i \in I} x_i \leq U$

3. **Integrality and Nonnegativity:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

#### Data Mapping

- $I$ (vehicle types):  
  All VehicleType values from capacity.csv (file_0_view_0, column VehicleType), and all ProductName values from products.csv (file_1_view_0, column ProductName).  
  Match by VehicleType = ProductName.

- $b_i$ (benefit coefficient):  
  products.csv, table_id: file_1_view_0, column Value, key: ProductName.

- $u_i$ (per-type limit):  
  capacity.csv, table_id: file_0_view_0, column Capacity, key: VehicleType.

- $U$ (total inventory capacity):  
  User-supplied or policy parameter (not present in the data; must be specified externally).

---

**Note:**  
- All vehicle types and their parameters must be matched by name between the two files.
- Every parameter and constraint is mapped directly to its source table and column as required.