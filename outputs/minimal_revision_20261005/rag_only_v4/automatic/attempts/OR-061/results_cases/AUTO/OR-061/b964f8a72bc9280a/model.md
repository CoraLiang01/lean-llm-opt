Mathematical Model

Sets:
- Let 𝑆 be the set of suppliers, indexed by i, where 𝑆 = {S1, S2, S3, S4, S5} (from file_1_view_0.Unnamed: 0).
- Let 𝐶 be the set of branches (customers), indexed by j, where 𝐶 = {C1, C2, C3, C4, C5} (from file_0_view_0.customer).

Parameters:
- f_i: Fixed cost of opening supplier i. Data Mapping: file_1_view_0.fixed_costs, indexed by i ∈ 𝑆.
- d_j: Demand at branch j. Data Mapping: file_0_view_0.demand, indexed by j ∈ 𝐶.
- c_{ij}: Transportation cost per unit from supplier i to branch j. Data Mapping: file_2_view_0, row Unnamed: 0 = i, column = j.

Decision Variables:
- y_i ∈ {0,1}: 1 if supplier i is open, 0 otherwise, ∀ i ∈ 𝑆.
- x_{ij} ≥ 0: Quantity supplied from supplier i to branch j, ∀ i ∈ 𝑆, j ∈ 𝐶.

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]
where:
- f_i: file_1_view_0.fixed_costs, i = file_1_view_0.Unnamed: 0
- c_{ij}: file_2_view_0, row Unnamed: 0 = i, column = j

Constraints:

1. Demand satisfaction at each branch:
\[
\sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
\]
where d_j: file_0_view_0.demand, j = file_0_view_0.customer

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in S, \forall j \in C
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in S
\]
\[
x_{ij} \geq 0 \quad \forall i \in S, \forall j \in C
\]

Data Mapping Summary:
- Suppliers S: file_1_view_0.Unnamed: 0
- Branches C: file_0_view_0.customer
- Fixed costs f_i: file_1_view_0.fixed_costs
- Demands d_j: file_0_view_0.demand
- Transportation costs c_{ij}: file_2_view_0, row Unnamed: 0 = i, column = j

This model determines which suppliers to open and how to allocate supply to branches to minimize total cost, using only the provided data.