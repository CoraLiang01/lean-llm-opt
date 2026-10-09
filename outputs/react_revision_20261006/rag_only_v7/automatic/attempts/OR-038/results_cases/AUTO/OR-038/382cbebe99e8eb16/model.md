Mathematical Model (Abstract Formulation):

Index Sets:
- 𝑉: Set of vehicle types, indexed by v (corresponding to VehicleType/ProductName in the data).

Parameters:
- cap_v: Daily inventory limit for vehicle type v. (from file_0_view_0, column Capacity)
- val_v: Benefit coefficient for vehicle type v. (from file_1_view_0, column Value)

Decision Variables:
- x_v: Number of units of vehicle type v to order per day (integer, x_v ≥ 0).

Objective:
Maximize total benefit:
\[
\max \sum_{v \in 𝑉} val_v \cdot x_v
\]

Subject to:
1. Per-vehicle-type daily inventory limits:
\[
0 \leq x_v \leq cap_v \quad \forall v \in 𝑉
\]
2. Total inventory capacity constraint:
\[
\sum_{v \in 𝑉} x_v \leq \sum_{v \in 𝑉} cap_v
\]
3. Integrality:
\[
x_v \in \mathbb{Z}_{\geq 0} \quad \forall v \in 𝑉
\]

Data Mapping:
- Index set 𝑉 is defined by all VehicleType values in file_0_view_0 and all ProductName values in file_1_view_0 (matched by name).
- cap_v: file_0_view_0, column Capacity, with key VehicleType = v.
- val_v: file_1_view_0, column Value, with key ProductName = v.
- Decision variable x_v corresponds to vehicle type v as above.