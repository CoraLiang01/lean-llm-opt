Mathematical Model

Sets:
- Let 𝑆 = {S₁, S₂, ..., S₈} be the set of suppliers, where each supplier Sᵢ is defined by Unnamed: 0 in table_id file_1_view_0 and file_2_view_0.
- Let 𝐶 = {C₁, C₂, ..., C₉} be the set of dealerships, where each dealership Cⱼ is defined by customer in table_id file_0_view_0 and by column names in file_2_view_0.

Parameters:
- fᵢ: Fixed cost of opening supplier Sᵢ, from fixed_costs in table_id file_1_view_0.
- dⱼ: Demand of dealership Cⱼ, from demand in table_id file_0_view_0.
- t_{ij}: Transportation cost per vehicle from supplier Sᵢ to dealership Cⱼ, from the (Sᵢ, Cⱼ) entry in table_id file_2_view_0.

Decision Variables:
- yᵢ ∈ {0,1}: 1 if supplier Sᵢ is open, 0 otherwise.
- x_{ij} ≥ 0: Number of vehicles supplied from Sᵢ to Cⱼ.

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction for each dealership:
\[
\sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
\]

2. Linking supply to open suppliers (no explicit supplier capacity is given, so only linking is enforced):
\[
x_{ij} \leq d_j y_i \quad \forall i \in S,\, j \in C
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in S
\]
\[
x_{ij} \geq 0 \quad \forall i \in S,\, j \in C
\]

Data Mapping

- S = {Unnamed: 0 | table_id: file_1_view_0}
- C = {customer | table_id: file_0_view_0}
- fᵢ = fixed_costs | table_id: file_1_view_0, indexed by Unnamed: 0
- dⱼ = demand | table_id: file_0_view_0, indexed by customer
- t_{ij} = (row: Unnamed: 0 = Sᵢ, column: Cⱼ) | table_id: file_2_view_0

Summary:
This is a capacitated facility location model with binary supplier opening variables and continuous shipment variables, using all suppliers and dealerships as defined in the provided CSVs. All parameters are mapped directly to the source data.