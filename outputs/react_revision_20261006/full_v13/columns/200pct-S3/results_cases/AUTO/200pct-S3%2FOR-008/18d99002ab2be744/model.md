ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, indexed by $i$. (From file_0_view_0.VehicleID and file_1_view_0.ProductName)

Parameters:
- $b_i$: Benefit coefficient of vehicle type $i$. (From file_1_view_0, column Value, matched by ProductName = VehicleType)
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$. (From file_0_view_0, column Capacity)
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable).

Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Constraints:
1. Vehicle-type-specific daily inventory limits:
\[
x_i \leq u_i, \quad \forall i \in I
\]
2. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

Data Mapping:
- $I$: All records in file_0_view_0.VehicleID (capacity.csv) and file_1_view_0.ProductName (products.csv), matched where file_0_view_0.VehicleType = file_1_view_0.ProductName.
- $b_i$: file_1_view_0.Value, matched by ProductName = VehicleType.
- $u_i$: file_0_view_0.Capacity, matched by VehicleType = ProductName.
- $x_i$: Decision variable for each $i \in I$.

No total inventory capacity constraint is present in the current data; only per-vehicle-type limits are enforced.