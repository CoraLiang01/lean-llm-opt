##### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit for vehicle type $i$

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Constraints:**
1. **Per-vehicle-type daily inventory limits:**
   \[
   0 \leq x_i \leq u_i \qquad \forall i \in I
   \]
2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

##### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv) and all ProductName values from file_1_view_0 (products.csv)
- $b_i$: Value column from file_1_view_0 (products.csv), mapped by ProductName = $i$
- $u_i$: Capacity column from file_0_view_0 (capacity.csv), mapped by VehicleType = $i$
- $x_i$: Decision variable for each $i \in I$ (vehicle type from above)

**Note:** The mapping between VehicleType (capacity.csv) and ProductName (products.csv) is by exact string match. All vehicle types present in both files are included in $I$.