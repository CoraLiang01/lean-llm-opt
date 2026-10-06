ABSTRACT MATHEMATICAL MODEL

Index Sets:
- Let $\mathcal{I}$ be the set of vehicle types, indexed by $i$.
  - Each $i \in \mathcal{I}$ is identified by (VehicleID, VehicleType) from capacity.csv and by ProductName from products.csv.

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Maximum daily inventory limit for vehicle type $i$.
- $U$: Total daily inventory capacity.

Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

Objective:
$$
\max \sum_{i \in \mathcal{I}} b_i x_i
$$

Subject to:
1. Per-vehicle-type daily inventory limits:
$$
x_i \leq u_i \quad \forall i \in \mathcal{I}
$$

2. Total daily inventory capacity:
$$
\sum_{i \in \mathcal{I}} x_i \leq U
$$

3. Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in \mathcal{I}
$$

DATA MAPPING

- $\mathcal{I}$: All records in capacity.csv (table_id: file_0_view_0), indexed by (VehicleID, VehicleType).
- $u_i$: Capacity column in capacity.csv (table_id: file_0_view_0), for each (VehicleID, VehicleType).
- $b_i$: Value column in products.csv (table_id: file_1_view_0), for each ProductName matching VehicleType.
- $U$: $\sum_{i \in \mathcal{I}} u_i$ (sum of Capacity column in capacity.csv, table_id: file_0_view_0).

Parameter Bindings:
- $u_i$: file_0_view_0, columns: VehicleID, VehicleType, Capacity
- $b_i$: file_1_view_0, columns: ProductName, Value

Each $x_i$ is indexed by (VehicleID, VehicleType) from capacity.csv, and $b_i$ is matched by VehicleType = ProductName.

No additional constraints or data are imposed beyond those described above.