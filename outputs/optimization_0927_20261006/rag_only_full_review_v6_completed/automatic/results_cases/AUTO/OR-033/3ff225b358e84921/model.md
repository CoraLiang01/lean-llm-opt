Abstract Optimization Model

Index Sets:
- Let \( I \) be the set of all products classified under ‘Baby’, indexed by \( i \).
 (Data Mapping: "Product Name" in table_id file_0_view_0, filtered by prefix "Baby")

Parameters:
- \( r_i \): Revenue per unit of product \( i \) ("Revenue" in file_0_view_0)
- \( s_i \): Initial inventory of product \( i \) ("Initial Inventory" in file_0_view_0)
- \( d_i \): Demand for product \( i \) ("Demand" in file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer

Objective:
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
- Index set \( I \): All records in "EuropeSalesRecords.csv" (table_id: file_0_view_0) where "Product Name" has prefix "Baby"
- \( r_i \): "Revenue" column in file_0_view_0
- \( s_i \): "Initial Inventory" column in file_0_view_0
- \( d_i \): "Demand" column in file_0_view_0

This model maximizes total revenue from fulfilling demand for ‘Baby’ products, subject to initial inventory and demand limits.