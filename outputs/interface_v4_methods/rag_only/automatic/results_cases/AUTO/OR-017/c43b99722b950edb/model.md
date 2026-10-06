Abstract Optimization Model

Index Sets:
- I: Set of all products with SKU prefix 'ZZ' (i ∈ I)

Parameters:
- r_i: Revenue per unit for product i (from Revenue column)
- s_i: Initial inventory for product i (from Initial Inventory column)
- d_i: Demand for product i (from Demand column)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

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

Variable Domains:
- x_i ∈ ℤ₊ (non-negative integers)

Data Mapping

- Index set I: All rows in RetailStoreSalesTransactions(ScannerData).csv where SKU (table_id: file_0_view_0, column: SKU) has prefix 'ZZ'
- Parameter r_i: Revenue (file_0_view_0, column: Revenue)
- Parameter s_i: Initial Inventory (file_0_view_0, column: Initial Inventory)
- Parameter d_i: Demand (file_0_view_0, column: Demand)
- Decision variable x_i: Number of units fulfilled for each i ∈ I

No literal data values or row counts are included; all mappings are symbolic and reference the exact table and column names.