**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data).

**Parameters:**
- $b_i$: Benefit coefficient for vehicle type $i$ (from products.csv, column Value, matched by VehicleType/ProductName).
- $u_i$: Maximum daily inventory limit for vehicle type $i$ (from capacity.csv, column Capacity, matched by VehicleType).
- $w_i$: Weight for vehicle type $i$ (from products.csv, column Weight, matched by VehicleType/ProductName).

**Decision Variables:**
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Vehicle-type-specific inventory limits:**
   \[
   x_i \leq u_i \qquad \forall i \in I
   \]
2. **Total inventory capacity constraint:**
   \[
   \sum_{i \in I} x_i \leq \sum_{i \in I} u_i
   \]
3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$ (vehicle types):  
  - `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv`, column `VehicleType`  
  - `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv`, column `ProductName`
- $b_i$ (benefit):  
  - `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv`, column `Value`, keyed by `ProductName`
- $u_i$ (inventory limit):  
  - `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv`, column `Capacity`, keyed by `VehicleType`
- $w_i$ (weight):  
  - `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv`, column `Weight`, keyed by `ProductName`

---

**Notes:**
- All vehicle types in both files are matched by name.
- The total inventory capacity is the sum of all per-type capacities, as implied by the constraint that the sum of all ordered units does not exceed the total inventory capacity.
- All variables are nonnegative integers, as required.