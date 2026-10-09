Mixed-Integer Fixed-Charge Transportation Model

Sets:
  I = {S1, S2, S3, S4, S5, S6}         // Suppliers (from supplier_capacity.csv)
  J = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12}   // Customers (from customer_demand.csv)

Parameters:
  SupplyCapacity_i      // Supply capacity of supplier i ∈ I (from supplier_capacity.csv, column SupplyCapacity)
  Demand_j              // Demand of customer j ∈ J (from customer_demand.csv, column Demand)
  VariableCost_ij       // Per-unit transportation cost from i to j (from route_variable_costs.csv, entry [i,j])
  FixedCost_ij          // Fixed activation cost for route i-j (from route_fixed_costs.csv, entry [i,j])
  M_ij                  // Big-M for linking x_ij and y_ij; set M_ij = SupplyCapacity_i

Decision Variables:
  x_ij ≥ 0              // Quantity shipped from supplier i to customer j (continuous)
  y_ij ∈ {0,1}          // 1 if route i-j is used, 0 otherwise (binary)

Objective:
  Minimize
    ∑_{i∈I} ∑_{j∈J} [ VariableCost_ij * x_ij + FixedCost_ij * y_ij ]

Subject to:
  // Supply capacity at each supplier
  ∑_{j∈J} x_ij ≤ SupplyCapacity_i        ∀ i ∈ I

  // Demand satisfaction at each customer
  ∑_{i∈I} x_ij = Demand_j                ∀ j ∈ J

  // Linking shipment and route activation
  x_ij ≤ SupplyCapacity_i * y_ij         ∀ i ∈ I, ∀ j ∈ J

  // Nonnegativity and binary restrictions
  x_ij ≥ 0                               ∀ i ∈ I, ∀ j ∈ J
  y_ij ∈ {0,1}                           ∀ i ∈ I, ∀ j ∈ J

Data Mapping:
- I: All Supplier values in supplier_capacity.csv (table_id: file_0_view_0, column: Supplier)
- J: All Customer values in customer_demand.csv (table_id: file_1_view_0, column: Customer)
- SupplyCapacity_i: supplier_capacity.csv (table_id: file_0_view_0, column: SupplyCapacity)
- Demand_j: customer_demand.csv (table_id: file_1_view_0, column: Demand)
- VariableCost_ij: route_variable_costs.csv (table_id: file_2_view_0, row: Supplier, column: Customer)
- FixedCost_ij: route_fixed_costs.csv (table_id: file_3_view_0, row: Supplier, column: Customer)
- M_ij: SupplyCapacity_i for each (i,j) pair

All indices, parameters, and variables are defined over the full set of suppliers and customers as listed in the current data.