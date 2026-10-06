ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i classified under category ‘ZZ’. (From RetailStoreSalesTransactions(ScannerData).csv, rows where SKU starts with 'ZZ')

Parameters:
- r_i: Revenue per unit for product i ∈ I. (Revenue column)
- s_i: Initial inventory for product i ∈ I. (Initial Inventory column)
- d_i: Demand quantity for product i ∈ I. (Demand column)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I
   \]
   (Each product’s fulfilled units cannot exceed its initial inventory or demand.)

Variable Domains:
- x_i ∈ ℤ_+ (non-negative integers)

Data Mapping:
- Index set I: All rows in table_id file_0_view_0 where SKU (column: SKU) has prefix 'ZZ'
- r_i: Revenue (column: Revenue, table_id: file_0_view_0)
- s_i: Initial Inventory (column: Initial Inventory, table_id: file_0_view_0)
- d_i: Demand (column: Demand, table_id: file_0_view_0)