Mathematical Model:

Sets:
- Let S be the set of suppliers, indexed by s. (S = {S1, S2, ..., S10}, from table_id: file_1_view_0, column: Unnamed: 0)
- Let C be the set of customer groups, indexed by c. (C = {C1, C2, ..., C10}, from table_id: file_0_view_0, column: customer)

Parameters:
- supply_capacity_s: Daily supply capacity of supplier s. (from table_id: file_1_view_0, column: supply_capacity)
- demand_c: Daily demand of customer group c. (from table_id: file_0_view_0, column: demand)
- cost_{s,c}: Transportation cost per unit from supplier s to customer group c. (from table_id: file_2_view_0, columns: C1...C10, rows: Unnamed: 0 = S1...S10)

Decision Variables:
- x_{s,c} ≥ 0: Number of units transported from supplier s to customer group c (continuous, non-negative).

Objective:
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]

Subject to:
1. Supply capacity constraints (for each supplier s):
\[
\sum_{c \in C} x_{s,c} \leq supply\_capacity_s \quad \forall s \in S
\]

2. Demand satisfaction constraints (for each customer group c):
\[
\sum_{s \in S} x_{s,c} = demand_c \quad \forall c \in C
\]

3. Non-negativity:
\[
x_{s,c} \geq 0 \quad \forall s \in S, c \in C
\]

Data Mapping:
- S = all Unnamed: 0 in table_id: file_1_view_0 (supplier IDs: S1...S10)
- C = all customer in table_id: file_0_view_0 (customer IDs: C1...C10)
- supply_capacity_s: table_id: file_1_view_0, column: supply_capacity, key: Unnamed: 0 = s
- demand_c: table_id: file_0_view_0, column: demand, key: customer = c
- cost_{s,c}: table_id: file_2_view_0, row: Unnamed: 0 = s, column: c

Variable Domain:
- x_{s,c} ≥ 0, continuous, ∀ s ∈ S, c ∈ C

All indices, parameters, and coefficients are bound exactly to the current data as specified above.