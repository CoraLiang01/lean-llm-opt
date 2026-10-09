ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products classified under ‘ELE-S’  
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference)

Parameters:
- r_i: Revenue per unit of product i ∈ I  
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory of product i ∈ I  
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ I  
  (Source: SalesStoreoverview.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill  
  Domain: Integer, 0 ≤ x_i ≤ min{s_i, d_i}

Objective:
- Maximize total revenue:
  max ∑_{i ∈ I} r_i x_i

Constraints:
1. Inventory constraint: x_i ≤ s_i  ∀ i ∈ I
2. Demand constraint:  x_i ≤ d_i  ∀ i ∈ I
3. Non-negativity:    x_i ≥ 0   ∀ i ∈ I
4. Integrality:      x_i ∈ ℤ   ∀ i ∈ I

Data Mapping:
- Index set I: file_0_view_0, column Product_Reference
- Parameter r_i: file_0_view_0, column Revenue
- Parameter s_i: file_0_view_0, column Initial Inventory
- Parameter d_i: file_0_view_0, column Demand

All data is sourced from SalesStoreoverview.csv, table_id: file_0_view_0.