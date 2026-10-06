Mathematical Model

Sets:
- Let 𝑆 = {S₁, S₂, ..., S₈} be the set of suppliers, where Sᵢ ∈ file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"].
- Let 𝐶 = {C₁, C₂, ..., C₉} be the set of dealerships, where Cⱼ ∈ file_0_view_0["customer"] and file_2_view_0 columns (excluding "Unnamed: 0").

Parameters:
- fᵢ: Fixed cost of opening supplier Sᵢ. Data Mapping: file_1_view_0["fixed_costs"] for Sᵢ.
- dⱼ: Demand of dealership Cⱼ. Data Mapping: file_0_view_0["demand"] for Cⱼ.
- t_{ij}: Transportation cost per vehicle from supplier Sᵢ to dealership Cⱼ. Data Mapping: file_2_view_0[Sᵢ, Cⱼ].

Decision Variables:
- yᵢ ∈ {0,1}: 1 if supplier Sᵢ is open, 0 otherwise.
- x_{ij} ≥ 0: Number of vehicles supplied from Sᵢ to Cⱼ.

Objective:
Minimize total cost (fixed + transportation):

\[
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} t_{ij} x_{ij}
\]

Constraints:

1. Demand satisfaction for each dealership:
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

Data Mapping Summary:
- S: file_1_view_0["Unnamed: 0"], file_2_view_0["Unnamed: 0"]
- C: file_0_view_0["customer"], file_2_view_0 columns ["C1", ..., "C9"]
- fᵢ: file_1_view_0["fixed_costs"] for Sᵢ
- dⱼ: file_0_view_0["demand"] for Cⱼ
- t_{ij}: file_2_view_0[Sᵢ, Cⱼ] (row Sᵢ, column Cⱼ)

This model determines which suppliers to open and how to allocate vehicle shipments to minimize total cost while meeting all dealership demands.