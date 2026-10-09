## Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit for vehicle type $i$

### Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

### Constraints:
1. **Vehicle-type daily inventory limits:**
   \[
   0 \leq x_i \leq u_i \qquad \forall i \in I
   \]
2. **Integrality:**
   \[
   x_i \in \mathbb{Z} \qquad \forall i \in I
   \]

### Data Mapping

- $I$: All VehicleType in file_0_view_0 (capacity.csv) and all ProductName in file_1_view_0 (products.csv)
- $b_i$: Value from file_1_view_0, column Value, for ProductName $i$
- $u_i$: Capacity from file_0_view_0, column Capacity, for VehicleType $i$
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

**Note:** The mapping between VehicleType (capacity.csv) and ProductName (products.csv) is by exact string match. Only vehicle types present in both files are included in $I$.