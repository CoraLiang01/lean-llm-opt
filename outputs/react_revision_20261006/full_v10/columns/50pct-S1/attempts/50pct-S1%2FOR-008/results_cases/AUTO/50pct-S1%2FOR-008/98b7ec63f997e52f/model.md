Mathematical Model

Index Sets:
- Let 𝑉 be the set of vehicle types, as identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
- Let b_i = Value for vehicle type i ∈ 𝑉, from file_1_view_0, column Value, where ProductName = i.
- Let u_i = Capacity for vehicle type i ∈ 𝑉, from file_0_view_0, column Capacity, where VehicleType = i.

Variables:
- For each i ∈ 𝑉, let x_i ∈ ℤ₊ be the number of vehicles of type i to order per day.

Objective:
Maximize total benefit:
$$
\max \sum_{i \in V} b_i x_i
$$

Constraints:
1. Vehicle-type daily inventory limits:
$$
x_i \leq u_i \quad \forall i \in V
$$

2. Total inventory capacity:
$$
\sum_{i \in V} x_i \leq \sum_{i \in V} u_i
$$

3. Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in V
$$

Data Mapping

- Index set V: All VehicleType values in file_0_view_0 and all ProductName values in file_1_view_0.
- Parameter b_i: file_1_view_0, column Value, key ProductName = i.
- Parameter u_i: file_0_view_0, column Capacity, key VehicleType = i.
- Variable x_i: number of vehicles of type i to order per day.
- All constraints and the objective use these parameters and index sets as defined above.