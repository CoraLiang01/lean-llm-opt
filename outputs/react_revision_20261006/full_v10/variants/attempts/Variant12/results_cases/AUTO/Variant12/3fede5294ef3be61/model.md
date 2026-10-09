Minimum-Cost Fixed-Charge Transportation Model

Sets:
  I: set of plants (from plant_capacity.csv), indexed by i ∈ I
  J: set of retailers (from retailer_demand.csv), indexed by j ∈ J

Parameters:
  SupplyCapacity_i: supply capacity of plant i (from plant_capacity.csv, column SupplyCapacity, table_id file_0_view_0)
  Demand_j: demand of retailer j (from retailer_demand.csv, column Demand, table_id file_1_view_0)
  c_ij: variable cost per carton from plant i to retailer j (from route_variable_costs.csv, table_id file_2_view_0, row Plant i, column j)
  f_ij: fixed cost to open route from plant i to retailer j (from route_fixed_costs.csv, table_id file_3_view_0, row Plant i, column j)
  M_ij: big-M for route (i,j), defined as M_ij = min(SupplyCapacity_i, Demand_j)

Decision Variables:
  x_ij ≥ 0: number of cartons shipped from plant i to retailer j (continuous)
  y_ij ∈ {0,1}: 1 if route (i,j) is opened, 0 otherwise (binary)

Objective:
  Minimize
    ∑_{i∈I} ∑_{j∈J} [ c_ij x_ij + f_ij y_ij ]

Subject to:
  1. Retailer demand satisfaction:
     ∑_{i∈I} x_ij = Demand_j  ∀ j ∈ J

  2. Plant supply capacity:
     ∑_{j∈J} x_ij ≤ SupplyCapacity_i  ∀ i ∈ I

  3. Route activation linking:
     x_ij ≤ M_ij y_ij  ∀ i ∈ I, j ∈ J

  4. Nonnegativity:
     x_ij ≥ 0  ∀ i ∈ I, j ∈ J

  5. Binary route activation:
     y_ij ∈ {0,1}  ∀ i ∈ I, j ∈ J

Data Mapping:
  - I = {P1, P2, P3}  (from plant_capacity.csv, table_id file_0_view_0, column Plant)
  - J = {R1, R2, R3, R4, R5, R6}  (from retailer_demand.csv, table_id file_1_view_0, column Retailer)
  - SupplyCapacity_i: from plant_capacity.csv, table_id file_0_view_0, column SupplyCapacity
  - Demand_j: from retailer_demand.csv, table_id file_1_view_0, column Demand
  - c_ij: from route_variable_costs.csv, table_id file_2_view_0, row Plant i, column j
  - f_ij: from route_fixed_costs.csv, table_id file_3_view_0, row Plant i, column j
  - M_ij = min(SupplyCapacity_i, Demand_j)  for each (i,j) pair

All sets, parameters, and variables are defined exactly as in the source data. No bounds are multiplied by y_ij except for the linking constraint. All constraints and the objective are as specified in the user request.