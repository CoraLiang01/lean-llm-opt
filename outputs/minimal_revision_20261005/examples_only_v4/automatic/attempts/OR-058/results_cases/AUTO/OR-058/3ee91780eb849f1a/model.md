Mathematical Model

Sets:
- Let 𝑆 be the set of suppliers, indexed by i. (𝑆 = all Unnamed: 0 in table_id file_1_view_0)
- Let 𝐶 be the set of stores, indexed by j. (𝐶 = all customer in table_id file_0_view_0)

Parameters:
- 𝑓ᵢ: Fixed cost of opening supplier i. (from fixed_costs in table_id file_1_view_0, indexed by Unnamed: 0)
- 𝑐_{ij}: Transportation cost per unit from supplier i to store j. (from table_id file_2_view_0, row Unnamed: 0 = i, column j = customer)
- 𝑑ⱼ: Demand at store j. (from demand in table_id file_0_view_0, indexed by customer)

Decision Variables:
- yᵢ ∈ {0,1}: 1 if supplier i is open, 0 otherwise.
- x_{ij} ≥ 0: Quantity supplied from supplier i to store j.

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each store:
\[
\sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
\]

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

Data Mapping

- S = {Unnamed: 0 | table_id: file_1_view_0}
- C = {customer | table_id: file_0_view_0}
- f_i = fixed_costs | table_id: file_1_view_0, indexed by Unnamed: 0
- c_{ij} = table_id: file_2_view_0, row Unnamed: 0 = i, column j = customer
- d_j = demand | table_id: file_0_view_0, indexed by customer