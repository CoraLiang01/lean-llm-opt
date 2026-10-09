Let:
- S = {S1, S2, ..., S18}: set of distribution centers, indexed by s, from file_1_view_0 column "Unnamed: 0"
- C = {C1, C2, ..., C18}: set of customer groups, indexed by c, from file_0_view_0 column "customer"
- d_c: demand of customer group c, from file_0_view_0 column "demand"
- u_s: supply capacity of distribution center s, from file_1_view_0 column "supply_capacity"
- cost_{s,c}: transportation cost per unit from s to c, from file_2_view_0, row "Unnamed: 0" (s), column c

Decision variables:
- x_{s,c} ≥ 0: quantity of goods transported from distribution center s to customer group c

Model:
Minimize total transportation cost:
  minimize ∑_{s∈S} ∑_{c∈C} cost_{s,c} · x_{s,c}

Subject to:
1. Demand satisfaction for each customer group:
  ∑_{s∈S} x_{s,c} = d_c   ∀ c ∈ C

2. Supply capacity for each distribution center:
  ∑_{c∈C} x_{s,c} ≤ u_s   ∀ s ∈ S

3. Nonnegativity:
  x_{s,c} ≥ 0   ∀ s ∈ S, c ∈ C

Data Mapping:
- S: All "Unnamed: 0" values in file_1_view_0 (supply_capacity.csv)
- C: All "customer" values in file_0_view_0 (customer_demand.csv)
- d_c: file_0_view_0, column "demand", indexed by "customer"
- u_s: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- cost_{s,c}: file_2_view_0, row "Unnamed: 0" (s), column c

Variable domains:
- x_{s,c} ∈ ℝ_+, ∀ s ∈ S, c ∈ C

Objective sense:
- Minimize total transportation cost

All indices, parameters, and coefficients are bound exactly as above to the current source data.