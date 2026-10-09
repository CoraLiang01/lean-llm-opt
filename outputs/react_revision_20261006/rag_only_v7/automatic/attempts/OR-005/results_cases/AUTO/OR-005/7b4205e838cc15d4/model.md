Mathematical Model (Transportation Problem):

Sets:
- Let S = {supplier1, supplier2, supplier3, supplier4, supplier5, supplier6, supplier7, supplier8}  (distribution centers, from file_1_view_0:Supplier)
- Let D = {demand1, demand2, demand3, demand4, demand5, demand6, demand7, demand8}  (customer groups, from file_0_view_0:Customers)

Parameters:
- demand_j: daily demand of customer group j ∈ D (from file_0_view_0:demand, indexed by Customers)
- supply_capacity_i: daily supply capacity of distribution center i ∈ S (from file_1_view_0:supply_capacity, indexed by Supplier)
- cost_{i,j}: transportation cost per unit from distribution center i ∈ S to customer group j ∈ D (from file_2_view_0, row Unnamed: 0 mapped to Supplier, columns demand1...demand8)

Decision Variables:
- x_{i,j} ≥ 0: quantity of goods shipped from distribution center i ∈ S to customer group j ∈ D

Objective:
Minimize total transportation cost:
\[
\min \sum_{i \in S} \sum_{j \in D} cost_{i,j} \cdot x_{i,j}
\]

Subject to:
1. Demand satisfaction for each customer group:
\[
\sum_{i \in S} x_{i,j} = demand_j \quad \forall j \in D
\]

2. Supply capacity for each distribution center:
\[
\sum_{j \in D} x_{i,j} \leq supply\_capacity_i \quad \forall i \in S
\]

3. Non-negativity:
\[
x_{i,j} \geq 0 \quad \forall i \in S, j \in D
\]

Data Mapping:
- S = all Supplier values in file_1_view_0:Supplier
- D = all Customers values in file_0_view_0:Customers
- demand_j = file_0_view_0:demand, indexed by Customers
- supply_capacity_i = file_1_view_0:supply_capacity, indexed by Supplier
- cost_{i,j} = file_2_view_0, with row Unnamed: 0 mapped to Supplier (using row_id_mapping), columns demand1...demand8 mapped to Customers

Variable Domains:
- x_{i,j} ≥ 0, continuous, ∀ i ∈ S, j ∈ D

All indices, parameters, and coefficients are bound exactly to the current source data as described above.