ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all “4U” products, indexed by i. (From OnlineSalesinUSA.csv, column: Product Name, filtered by prefix "4U")

Parameters:
- r_i: Revenue per unit for product i ∈ I. (OnlineSalesinUSA.csv, column: Revenue)
- d_i: Demand for product i ∈ I during the sales horizon. (OnlineSalesinUSA.csv, column: Demand)
- s_i: Initial inventory available for product i ∈ I. (OnlineSalesinUSA.csv, column: Initial Inventory)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue from fulfilled sales:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint (cannot fulfill more than available inventory):
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint (cannot fulfill more than demand):
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in OnlineSalesinUSA.csv where Product Name starts with "4U" (table_id: file_0_view_0, column: Product Name)
- Parameter r_i: OnlineSalesinUSA.csv, column: Revenue, table_id: file_0_view_0
- Parameter d_i: OnlineSalesinUSA.csv, column: Demand, table_id: file_0_view_0
- Parameter s_i: OnlineSalesinUSA.csv, column: Initial Inventory, table_id: file_0_view_0

This model maximizes total revenue from “4U” products by optimally allocating fulfillment quantities, subject to both inventory and demand limits, as specified in the provided data.