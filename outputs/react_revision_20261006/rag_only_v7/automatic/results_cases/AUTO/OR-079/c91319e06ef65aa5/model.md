Mathematical Model (Uncapacitated Facility Location Problem with Capacities):

Sets:
- Let F be the set of potential factory sites (facilities): F = {A1, A2, ..., A15}, from file_0_view_0["Facility"].
- Let D be the set of distribution centers: D = {B1, B2, ..., B8}, from file_2_view_0["Destination"].

Parameters:
- FixedCost_f: Fixed cost of opening facility f ∈ F, from file_0_view_0["FixedCost"].
- Capacity_f: Maximum capacity of facility f ∈ F, from file_0_view_0["Capacity"].
- Demand_d: Demand at distribution center d ∈ D, from file_2_view_0["Demand"].
- ShipCost_{f,d}: Per-unit shipping cost from facility f ∈ F to distribution center d ∈ D, from file_1_view_0, with row "Origin" = f and column d.

Decision Variables:
- y_f ∈ {0,1}: 1 if facility f ∈ F is constructed, 0 otherwise.
- x_{f,d} ≥ 0: Quantity shipped from facility f ∈ F to distribution center d ∈ D.

Objective:
Minimize total cost:
\[
\min \sum_{f \in F} \text{FixedCost}_f \cdot y_f + \sum_{f \in F} \sum_{d \in D} \text{ShipCost}_{f,d} \cdot x_{f,d}
\]

Subject to:
1. Demand satisfaction at each distribution center:
\[
\forall d \in D: \quad \sum_{f \in F} x_{f,d} = \text{Demand}_d
\]

2. Facility capacity (unconditional, as per data and instructions):
\[
\forall f \in F: \quad \sum_{d \in D} x_{f,d} \leq \text{Capacity}_f
\]

3. Facility activation:
\[
\forall f \in F, \forall d \in D: \quad x_{f,d} \geq 0
\]
\[
\forall f \in F: \quad y_f \in \{0,1\}
\]

Data Mapping:
- F = file_0_view_0["Facility"]
- D = file_2_view_0["Destination"]
- FixedCost_f: file_0_view_0["FixedCost"], indexed by Facility = f
- Capacity_f: file_0_view_0["Capacity"], indexed by Facility = f
- Demand_d: file_2_view_0["Demand"], indexed by Destination = d
- ShipCost_{f,d}: file_1_view_0, row "Origin" = f, column d

All parameters and index sets are bound exactly to the current CSV data as described above.