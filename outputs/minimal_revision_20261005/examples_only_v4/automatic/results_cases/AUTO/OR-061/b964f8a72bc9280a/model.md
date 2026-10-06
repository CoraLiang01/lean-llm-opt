Mathematical Model

Sets:
- Let 𝑰 be the set of suppliers, indexed by i, where 𝑰 = {S1, S2, S3, S4, S5} (from file_1_view_0."Unnamed: 0").
- Let 𝑱 be the set of branches, indexed by j, where 𝑱 = {C1, C2, C3, C4, C5} (from file_0_view_0."customer").

Parameters:
- fᵢ: Fixed cost of opening supplier i. Data Mapping: file_1_view_0."fixed_costs" for supplier i = file_1_view_0."Unnamed: 0".
- dⱼ: Demand at branch j. Data Mapping: file_0_view_0."demand" for branch j = file_0_view_0."customer".
- c_{ij}: Transportation cost per unit from supplier i to branch j. Data Mapping: file_2_view_0, row i = file_2_view_0."Unnamed: 0", column j = file_2_view_0."Ck" where k = j.

Decision Variables:
- yᵢ ∈ {0,1}: 1 if supplier i is open, 0 otherwise, ∀i ∈ 𝑰.
- x_{ij} ≥ 0: Quantity supplied from supplier i to branch j, ∀i ∈ 𝑰, ∀j ∈ 𝑱.

Objective:
Minimize total cost (fixed + transportation):
\[
\min \sum_{i \in 𝑰} f_i y_i + \sum_{i \in 𝑰} \sum_{j \in 𝑱} c_{ij} x_{ij}
\]

Subject to:

1. Demand satisfaction at each branch:
\[
\sum_{i \in 𝑰} x_{ij} = d_j \quad \forall j \in 𝑱
\]

2. Supply only from open suppliers:
\[
x_{ij} \leq d_j y_i \quad \forall i \in 𝑰, \forall j \in 𝑱
\]

3. Variable domains:
\[
y_i \in \{0,1\} \quad \forall i \in 𝑰
\]
\[
x_{ij} \geq 0 \quad \forall i \in 𝑰, \forall j \in 𝑱
\]

Data Mapping Summary:
- 𝑰: file_1_view_0."Unnamed: 0"
- 𝑱: file_0_view_0."customer"
- fᵢ: file_1_view_0."fixed_costs" (supplier i)
- dⱼ: file_0_view_0."demand" (branch j)
- c_{ij}: file_2_view_0, row i = file_2_view_0."Unnamed: 0", column j = file_2_view_0."Ck" where k = j

This model determines which suppliers to open and how much each should supply to each branch to minimize total costs while meeting all branch demands.