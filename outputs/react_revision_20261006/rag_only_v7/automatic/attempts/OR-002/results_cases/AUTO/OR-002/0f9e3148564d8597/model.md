Mathematical Model (Transportation Problem):

Sets:
- S: Set of Walmart stores (indexed by s), from file_1_view_0["Unnamed: 0"]
- C: Set of customer groups (indexed by c), from file_0_view_0["customer"]

Parameters:
- demand_c: Demand of customer group c ∈ C, from file_0_view_0["demand"]
- supply_capacity_s: Supply capacity of store s ∈ S, from file_1_view_0["supply_capacity"]
- cost_sc: Transportation cost per unit from store s to customer group c, from file_2_view_0, row "Unnamed: 0" = s, column c

Decision Variables:
- x_sc: Quantity transported from store s ∈ S to customer group c ∈ C, x_sc ≥ 0 (continuous)

Objective:
Minimize total transportation cost:
minimize ∑_{s∈S} ∑_{c∈C} cost_sc · x_sc

Subject to:
1. Demand satisfaction for each customer group:
  ∑_{s∈S} x_sc = demand_c  ∀ c ∈ C

2. Supply capacity for each store:
  ∑_{c∈C} x_sc ≤ supply_capacity_s  ∀ s ∈ S

3. Non-negativity:
  x_sc ≥ 0  ∀ s ∈ S, c ∈ C

Data Mapping:
- S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11}  (table_id: file_1_view_0, column: "Unnamed: 0")
- C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12}  (table_id: file_0_view_0, column: "customer")
- demand_c: file_0_view_0["demand"], indexed by c ∈ C
- supply_capacity_s: file_1_view_0["supply_capacity"], indexed by s ∈ S
- cost_sc: file_2_view_0, row "Unnamed: 0" = s, column c

Variables:
- x_sc ≥ 0, ∀ s ∈ S, c ∈ C

Objective:
minimize ∑_{s∈S} ∑_{c∈C} cost_sc · x_sc

Constraints:
1. ∑_{s∈S} x_sc = demand_c  ∀ c ∈ C
2. ∑_{c∈C} x_sc ≤ supply_capacity_s  ∀ s ∈ S
3. x_sc ≥ 0  ∀ s ∈ S, c ∈ C