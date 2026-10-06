ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified under ‘ZZ’. (I = {i : SKU of product i starts with 'ZZ'} from table_id: file_0_view_0, column: SKU)

Parameters:
- r_i: Revenue per unit for product i ∈ I. (table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory for product i ∈ I. (table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ I. (table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
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
- Index set I: All rows in table_id: file_0_view_0 where SKU starts with 'ZZ' (column: SKU, file: RetailStoreSalesTransactions(ScannerData).csv)
- Parameter r_i: Revenue (column: Revenue, table_id: file_0_view_0)
- Parameter s_i: Initial Inventory (column: Initial Inventory, table_id: file_0_view_0)
- Parameter d_i: Demand (column: Demand, table_id: file_0_view_0)

No literal data values or record counts are included; all data is referenced symbolically by table and column.