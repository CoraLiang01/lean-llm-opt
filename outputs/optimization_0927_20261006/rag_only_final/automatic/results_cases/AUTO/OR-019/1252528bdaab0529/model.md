ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified as ‘27in’. (From SalesDataAnalysis.csv, table_id: file_0_view_0, column: Product Name, filtered to include only '27in' products.)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \). (file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \). (file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, for all \( i \in I \).
 Domain: \( x_i \geq 0 \), integer or continuous as appropriate.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory constraint: \( x_i \leq s_i \) for all \( i \in I \)
2. Demand constraint:  \( x_i \leq d_i \) for all \( i \in I \)
3. Nonnegativity:    \( x_i \geq 0 \) for all \( i \in I \)

Data Mapping:
- Index set \( I \): All rows in SalesDataAnalysis.csv (table_id: file_0_view_0) where Product Name contains '27in'
- \( r_i \): Revenue (file_0_view_0, column: Revenue)
- \( d_i \): Demand (file_0_view_0, column: Demand)
- \( s_i \): Initial Inventory (file_0_view_0, column: Initial Inventory)
- Decision variable \( x_i \): Number of units fulfilled for each product \( i \in I \)

No data values or record counts are included; all sets and parameters are defined symbolically and mapped to their exact sources.