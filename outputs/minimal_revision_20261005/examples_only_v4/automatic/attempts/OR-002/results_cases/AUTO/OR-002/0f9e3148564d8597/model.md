Mathematical Optimization Model

Sets:
- S: set of Walmart stores (indexed by s), S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11} [from file_1_view_0.Unnamed: 0]
- C: set of customer groups (indexed by c), C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} [from file_0_view_0.customer]

Parameters:
- demand_c = demand of customer group c [from file_0_view_0.demand]
- supply_s = supply capacity of store s [from file_1_view_0.supply_capacity]
- cost_sc = transportation cost per unit from store s to customer group c [from file_2_view_0, row Unnamed: 0 = s, column c]

Decision Variables:
- x_sc = quantity transported from store s to customer group c ∀ s ∈ S, c ∈ C  [x_sc ≥ 0, continuous]

Objective:
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} \text{cost}_{sc} \cdot x_{sc}
\]

Subject to:

1. Demand satisfaction for each customer group:
\[
\sum_{s \in S} x_{sc} = \text{demand}_c \qquad \forall c \in C
\]

2. Supply capacity for each store:
\[
\sum_{c \in C} x_{sc} \leq \text{supply}_s \qquad \forall s \in S
\]

3. Non-negativity:
\[
x_{sc} \geq 0 \qquad \forall s \in S,\, c \in C
\]

Data Mapping

- S (stores): file_1_view_0.Unnamed: 0
- C (customers): file_0_view_0.customer
- demand_c: file_0_view_0.demand, indexed by customer
- supply_s: file_1_view_0.supply_capacity, indexed by Unnamed: 0
- cost_sc: file_2_view_0, row Unnamed: 0 = s, column = c

All indices, parameters, and coefficients are bound directly to the supplied data as specified above.