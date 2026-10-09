Mathematical Model:

Sets:
- Let F be the set of suppliers, indexed by i. (F = {S1, S2, S3, S4, S5, S6, S7, S8}, from file_1_view_0["Unnamed: 0"])
- Let D be the set of dealerships, indexed by j. (D = {C1, C2, C3, C4, C5, C6, C7, C8, C9}, from file_0_view_0["customer"])

Parameters:
- demand_j: Demand of dealership j ∈ D. (from file_0_view_0["demand"])
- fixed_cost_i: Fixed cost to open supplier i ∈ F. (from file_1_view_0["fixed_costs"])
- trans_cost_{ij}: Transportation cost per vehicle from supplier i ∈ F to dealership j ∈ D. (from file_2_view_0, row "Unnamed: 0" = i, column = j)

Decision Variables:
- y_i ∈ {0,1}: 1 if supplier i ∈ F is open, 0 otherwise.
- x_{ij} ≥ 0: Number of vehicles supplied from supplier i ∈ F to dealership j ∈ D.

Objective:
Minimize total cost:
\[
\min \sum_{i \in F} fixed\_cost_i \cdot y_i + \sum_{i \in F} \sum_{j \in D} trans\_cost_{ij} \cdot x_{ij}
\]

Subject to:
1. Demand satisfaction at each dealership:
\[
\forall j \in D: \quad \sum_{i \in F} x_{ij} = demand_j
\]

2. Supply only from open suppliers:
\[
\forall i \in F, \forall j \in D: \quad x_{ij} \leq demand_j \cdot y_i
\]

3. Variable domains:
\[
\forall i \in F: \quad y_i \in \{0,1\}
\]
\[
\forall i \in F, \forall j \in D: \quad x_{ij} \geq 0
\]

Data Mapping:
- F (suppliers): file_1_view_0["Unnamed: 0"]
- D (dealerships): file_0_view_0["customer"]
- demand_j: file_0_view_0["demand"], indexed by j ∈ D
- fixed_cost_i: file_1_view_0["fixed_costs"], indexed by i ∈ F
- trans_cost_{ij}: file_2_view_0, row "Unnamed: 0" = i ∈ F, column = j ∈ D

All indices, parameters, and constraints are mapped directly to the provided CSV data.