#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of vehicle types (from capacity.csv: VehicleType, and products.csv: ProductName)

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$  
- $u_i$: Maximum daily inventory limit for vehicle type $i$  
- $U$: Total daily inventory capacity

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-type inventory limits:**  
   \[
   x_i \leq u_i \quad \forall i \in I
   \]
2. **Total inventory capacity:**  
   \[
   \sum_{i \in I} x_i \leq U
   \]
3. **Integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$:  
  All VehicleType values from file_0_view_0 (capacity.csv, column VehicleType)  
  All ProductName values from file_1_view_0 (products.csv, column ProductName)  
  (Align $i$ by matching VehicleType = ProductName)

- $b_i$:  
  file_1_view_0 (products.csv), column Value, keyed by ProductName

- $u_i$:  
  file_0_view_0 (capacity.csv), column Capacity, keyed by VehicleType

- $U$:  
  $U = \sum_{i \in I} u_i$  
  (Sum of all Capacity values in file_0_view_0, capacity.csv)

- $x_i$:  
  Decision variable for each $i \in I$

---

**Note:**  
- All parameters and index sets are mapped directly from the original CSV files using the exact column names and file/table IDs as above.
- The model maximizes total benefit from daily vehicle orders, subject to both per-type and total inventory limits, with integer decision variables.