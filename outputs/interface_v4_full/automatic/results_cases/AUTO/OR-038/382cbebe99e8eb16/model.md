## Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (with business key VehicleType from file_0_view_0 and ProductName from file_1_view_0)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from Value in file_1_view_0)
- $u_i$ = daily inventory limit for vehicle type $i$ (from Capacity in file_0_view_0)
- $U$ = total inventory capacity per day (sum of all $x_i$)

### Objective
$$
\max \sum_{i \in I} b_i x_i
$$

### Constraints

1. **Per-type daily inventory limits:**
   $$
   x_i \leq u_i, \quad \forall i \in I
   $$

2. **Total inventory capacity:**
   $$
   \sum_{i \in I} x_i \leq U
   $$
   (Here, $U$ is a given total inventory capacity parameter, which must be provided externally or set as the sum of all $u_i$ if not otherwise specified.)

3. **Integrality and nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

## Data Mapping

- $i \in I$ (vehicle types):  
  - file_0_view_0.VehicleType  
  - file_1_view_0.ProductName  
  (joined on VehicleType = ProductName)

- $b_i$ (benefit coefficient):  
  - file_1_view_0.Value (for ProductName = $i$)

- $u_i$ (per-type daily inventory limit):  
  - file_0_view_0.Capacity (for VehicleType = $i$)

- $U$ (total inventory capacity):  
  - **[User must specify, or set $U = \sum_{i \in I} u_i$ if not otherwise provided]**

---

## Variable Domains

- $x_i$ are nonnegative integers for all $i \in I$.

---

## Source Tables Used

- file_0_view_0: capacity.csv (columns: VehicleType, Capacity)
- file_1_view_0: products.csv (columns: ProductName, Value)