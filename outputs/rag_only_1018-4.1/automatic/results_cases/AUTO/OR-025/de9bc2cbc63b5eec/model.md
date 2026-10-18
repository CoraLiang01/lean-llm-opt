ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all tablet smartphone models,  
  I = {i : 'Product Name' in SmartphoneRetailOutletSalesData.csv where 'Product Name' starts with 'TABLET'}

Parameters:
- r_i: Revenue per unit for model i  
  (Source: SmartphoneRetailOutletSalesData.csv, column 'Revenue', table_id: file_0_view_0)
- inv_i: Initial inventory for model i  
  (Source: SmartphoneRetailOutletSalesData.csv, column 'Initial Inventory', table_id: file_0_view_0)
- d_i: Demand for model i  
  (Source: SmartphoneRetailOutletSalesData.csv, column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of model i to fulfill  
  Domain: integer, x_i ≥ 0

Objective:
- Maximize total revenue:
  maximize Z = ∑_{i ∈ I} r_i * x_i

Constraints:
1. Inventory constraint for each model:
  x_i ≤ inv_i  ∀ i ∈ I

2. Demand constraint for each model:
  x_i ≤ d_i  ∀ i ∈ I

3. Non-negativity and integrality:
  x_i ∈ {0, 1, 2, ...}  ∀ i ∈ I

Data Mapping:
- Index set I, and all parameters (r_i, inv_i, d_i) are sourced from SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, using only rows where 'Product Name' starts with 'TABLET' (predicate: 'Product Name' prefix 'TABLET').  
- 'Revenue' → r_i  
- 'Initial Inventory' → inv_i  
- 'Demand' → d_i  
- 'Product Name' → model index i

No additional constraints or modifications are imposed beyond those specified in the user query and the validated data filter.