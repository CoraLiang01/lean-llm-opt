ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all smartphone models classified as 'TABLET'  
  (Source: SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit for model i ∈ I  
  (Source: SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory for model i ∈ I  
  (Source: SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for model i ∈ I  
  (Source: SmartphoneRetailOutletSalesData.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of model i ∈ I to fulfill  
  Domain: Integer, 0 ≤ x_i ≤ min{s_i, d_i}

Objective:
- Maximize total revenue:
  maximize Z = ∑_{i ∈ I} r_i x_i

Constraints:
1. Inventory constraint: x_i ≤ s_i  ∀ i ∈ I
2. Demand constraint:  x_i ≤ d_i  ∀ i ∈ I
3. Non-negativity and integrality: x_i ∈ {0, 1, ..., min{s_i, d_i}} ∀ i ∈ I

Data Mapping:
- Index set I, and all parameters (r_i, s_i, d_i) are sourced from SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0), using:
  - Product Name (for I)
  - Revenue (for r_i)
  - Initial Inventory (for s_i)
  - Demand (for d_i)
- Only rows where Product Name has prefix 'TABLET' are included.