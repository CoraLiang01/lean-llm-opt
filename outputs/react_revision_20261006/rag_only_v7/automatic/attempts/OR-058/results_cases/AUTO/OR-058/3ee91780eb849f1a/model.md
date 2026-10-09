Mathematical Model:

Sets:
- Let F be the set of suppliers, indexed by i, with F = {S1, S2, S3, S4, S5, S6} (from file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0).
- Let C be the set of stores, indexed by j, with C = {C1, C2, C3, C4, C5, C6} (from file_0_view_0.customer and file_2_view_0 columns).

Parameters:
- fixed_cost_i: Fixed cost to open supplier i, from file_1_view_0.fixed_costs, for i ∈ F.
- demand_j: Demand at store j, from file_0_view_0.demand, for j ∈ C.
- trans_cost_{ij}: Transportation cost per unit from supplier i to store j, from file_2_view_0, for (i, j) ∈ F × C.

Decision Variables:
- y_i ∈ {0,1}: 1 if supplier i is open, 0 otherwise, for i ∈ F.
- x_{ij} ≥ 0: Quantity supplied from supplier i to store j, for (i, j) ∈ F × C.

Objective:
Minimize total cost:
\[
\min \sum_{i \in F} \text{fixed_cost}_i \cdot y_i + \sum_{i \in F} \sum_{j \in C} \text{trans_cost}_{ij} \cdot x_{ij}
\]

Subject to:
1. Demand satisfaction at each store:
\[
\sum_{i \in F} x_{ij} = \text{demand}_j \quad \forall j \in C
\]
2. Supply only from open suppliers:
\[
x_{ij} \leq \text{demand}_j \cdot y_i \quad \forall i \in F, \forall j \in C
\]
3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
\]

Data Mapping:
- F (suppliers): file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0
- C (stores): file_0_view_0.customer and file_2_view_0 columns C1–C6
- fixed_cost_i: file_1_view_0.fixed_costs, mapped by Unnamed: 0 = i
- demand_j: file_0_view_0.demand, mapped by customer = j
- trans_cost_{ij}: file_2_view_0, entry at row Unnamed: 0 = i, column j

Variables:
- y_i: binary, for i ∈ F
- x_{ij}: continuous ≥ 0, for (i, j) ∈ F × C

Constraints:
- Demand satisfaction: sum over i of x_{ij} = demand_j for each j ∈ C (file_0_view_0)
- Supply only from open suppliers: x_{ij} ≤ demand_j * y_i for all (i, j)
- Domains: y_i ∈ {0,1}, x_{ij} ≥ 0

Objective:
- Minimize total cost: sum over i of fixed_cost_i * y_i (file_1_view_0) plus sum over (i, j) of trans_cost_{ij} * x_{ij} (file_2_view_0)

All index sets, parameters, and mappings are defined exactly as per the current CSV data.