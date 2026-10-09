Mathematical Model:

Sets:
- Let F be the set of warehouses, indexed by i. (F = {S1, S2, S3}, from file_1_view_0["Unnamed: 0"])
- Let C be the set of musicians/bands (customers), indexed by j. (C = {C1, C2, C3}, from file_0_view_0["customer"])

Parameters:
- d_j: Demand of customer j. (from file_0_view_0["demand"])
- f_i: Fixed cost to open warehouse i. (from file_1_view_0["fixed_costs"])
- t_{ij}: Transportation cost per unit from warehouse i to customer j. (from file_2_view_0, row "Unnamed: 0" = i, column = j)

Decision Variables:
- y_i ∈ {0,1}: 1 if warehouse i is opened, 0 otherwise.
- x_{ij} ≥ 0: Quantity of goods supplied from warehouse i to customer j.

Objective:
Minimize total cost:
\[
\min \sum_{i \in F} f_i y_i + \sum_{i \in F} \sum_{j \in C} t_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction for each customer:
\[
\sum_{i \in F} x_{ij} = d_j \quad \forall j \in C
\]

2. Supply from a warehouse only if it is open:
\[
x_{ij} \leq d_j y_i \quad \forall i \in F, \forall j \in C
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in F
\]
\[
x_{ij} \geq 0 \quad \forall i \in F, \forall j \in C
\]

Data Mapping:
- F (warehouses): file_1_view_0["Unnamed: 0"]
- C (customers): file_0_view_0["customer"]
- d_j: file_0_view_0["demand"], indexed by customer j
- f_i: file_1_view_0["fixed_costs"], indexed by warehouse i
- t_{ij}: file_2_view_0, row "Unnamed: 0" = i, column = j (customer)

All indices, parameters, and constraints are mapped directly to the provided CSV data.