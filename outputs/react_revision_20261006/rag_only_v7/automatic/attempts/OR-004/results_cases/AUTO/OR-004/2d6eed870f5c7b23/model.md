Mathematical Model:

Sets:
- Let S be the set of distribution centers, indexed by s, with S = {S1, S2, ..., S12} (from file_1_view_0, column "Unnamed: 0").
- Let C be the set of customer groups, indexed by c, with C = {C1, C2, ..., C12} (from file_0_view_0, column "customer").

Parameters:
- demand_c: Demand of customer group c ∈ C. (from file_0_view_0, column "demand")
- supply_capacity_s: Supply capacity of distribution center s ∈ S. (from file_1_view_0, column "supply_capacity")
- cost_sc: Transportation cost per unit from distribution center s ∈ S to customer group c ∈ C. (from file_2_view_0, columns "C1"..."C12", rows "Unnamed: 0" = S1...S12)

Decision Variables:
- x_sc ≥ 0: Number of units shipped from distribution center s ∈ S to customer group c ∈ C.

Objective:
Minimize total transportation cost:
minimize ∑_{s∈S} ∑_{c∈C} cost_sc × x_sc

Subject to:
1. Demand satisfaction for each customer group:
  ∑_{s∈S} x_sc = demand_c  ∀ c ∈ C

2. Supply capacity for each distribution center:
  ∑_{c∈C} x_sc ≤ supply_capacity_s  ∀ s ∈ S

3. Non-negativity:
  x_sc ≥ 0  ∀ s ∈ S, c ∈ C

Data Mapping:
- S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12} from file_1_view_0, column "Unnamed: 0"
- C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} from file_0_view_0, column "customer"
- demand_c: file_0_view_0, column "demand", indexed by "customer"
- supply_capacity_s: file_1_view_0, column "supply_capacity", indexed by "Unnamed: 0"
- cost_sc: file_2_view_0, value at row "Unnamed: 0" = s, column = c

Variable domains:
- x_sc ≥ 0, continuous, for all s ∈ S, c ∈ C

All indices, parameters, and coefficients are bound directly to the current source data as specified.