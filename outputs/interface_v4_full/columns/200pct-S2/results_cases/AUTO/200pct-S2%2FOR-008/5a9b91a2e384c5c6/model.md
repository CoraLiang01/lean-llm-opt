#### Abstract Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (with business identifier VehicleType from file_0_view_0 and ProductName from file_1_view_0)
- $x_i$ = number of vehicles of type $i$ to order per day (integer decision variable)

Parameters:
- $b_i$ = benefit coefficient for vehicle type $i$ (from Value in file_1_view_0, matched by ProductName = VehicleType)
- $u_i$ = daily inventory limit for vehicle type $i$ (from Capacity in file_0_view_0)
- $C$ = total inventory capacity per day (sum of all $x_i$)

#### Objective
$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints

1. **Total Inventory Capacity Constraint**
   $$
   \sum_{i \in I} x_i \leq C
   $$
   (Here, $C$ is the total inventory capacity per day, as specified in the user description.)

2. **Vehicle Type-Specific Inventory Limits**
   $$
   x_i \leq u_i \quad \forall i \in I
   $$

3. **Nonnegativity and Integrality**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

#### Data Mapping

- $I$ (vehicle types): file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name)
- $b_i$: file_1_view_0.Value, with $i$ identified by ProductName
- $u_i$: file_0_view_0.Capacity, with $i$ identified by VehicleType
- $x_i$: decision variable for each $i \in I$

- $C$: total inventory capacity per day (user-supplied; if not specified in data, must be provided as a parameter)

#### Notes

- All parameters and identifiers are preserved as in the source files.
- Each $x_i$ is a nonnegative integer.
- The model maximizes total benefit from daily vehicle orders, subject to both total and type-specific inventory limits.