Mathematical Model

Sets:
- Let 𝑆 be the set of suppliers, indexed by i, where 𝑆 = {S1, S2, ..., S24} (from file_1_view_0, column "Unnamed: 0" and file_2_view_0, row "Unnamed: 0").
- Let 𝐶 be the set of supermarkets, indexed by j, where 𝐶 = {C1, C2, ..., C25} (from file_0_view_0, column "customer" and file_2_view_0, columns C1–C25).

Parameters:
- 𝑓ᵢ: Fixed cost of opening supplier i.  
  Data Mapping: file_1_view_0, columns "Unnamed: 0" (supplier index), "fixed_costs"
- 𝑐ᵢⱼ: Transportation cost per unit from supplier i to supermarket j.  
  Data Mapping: file_2_view_0, rows "Unnamed: 0" (supplier index), columns C1–C25 (supermarket index)
- 𝑑ⱼ: Demand of supermarket j.  
  Data Mapping: file_0_view_0, columns "customer" (supermarket index), "demand"

Decision Variables:
- 𝑦ᵢ ∈ {0,1}: 1 if supplier i is open, 0 otherwise, ∀i ∈ 𝑆
- 𝑥ᵢⱼ ≥ 0: Quantity supplied from supplier i to supermarket j, ∀i ∈ 𝑆, ∀j ∈ 𝐶

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in S} f_i y_i + \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction for each supermarket:
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
- 𝑆 (suppliers): file_1_view_0, "Unnamed: 0"; file_2_view_0, "Unnamed: 0"
- 𝐶 (supermarkets): file_0_view_0, "customer"; file_2_view_0, columns C1–C25
- 𝑓ᵢ: file_1_view_0, "fixed_costs"
- 𝑐ᵢⱼ: file_2_view_0, rows "Unnamed: 0", columns C1–C25
- 𝑑ⱼ: file_0_view_0, "demand"

All indices and parameters are defined over the full set of entities present in the respective CSV files.