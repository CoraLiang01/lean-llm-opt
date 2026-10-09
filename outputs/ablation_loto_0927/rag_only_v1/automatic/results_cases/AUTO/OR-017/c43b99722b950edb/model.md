ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i where SKU starts with 'ZZ' (i.e., all 'ZZ' products).

Parameters:
- r_i: Revenue per unit of product i. Source: RetailStoreSalesTransactions(ScannerData).csv, column 'Revenue', table_id: file_0_view_0.
- s_i: Initial inventory of product i. Source: RetailStoreSalesTransactions(ScannerData).csv, column 'Initial Inventory', table_id: file_0_view_0.
- d_i: Demand for product i. Source: RetailStoreSalesTransactions(ScannerData).csv, column 'Demand', table_id: file_0_view_0.

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i}), ∀ i ∈ I.

Objective:
- Maximize total revenue from 'ZZ' products:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in RetailStoreSalesTransactions(ScannerData).csv where 'SKU' (table_id: file_0_view_0, column 'SKU') has prefix 'ZZ'.
- Parameter r_i: 'Revenue' column, table_id: file_0_view_0.
- Parameter s_i: 'Initial Inventory' column, table_id: file_0_view_0.
- Parameter d_i: 'Demand' column, table_id: file_0_view_0.