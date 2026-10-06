ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under ‘Baby’ (from EuropeSalesRecords.csv where Product Name has prefix 'Baby').

Parameters:
- \( r_i \): Revenue per unit of product \( i \in I \) (EuropeSalesRecords.csv, column: Revenue).
- \( d_i \): Demand quantity for product \( i \in I \) (EuropeSalesRecords.csv, column: Demand).
- \( s_i \): Initial inventory for product \( i \in I \) (EuropeSalesRecords.csv, column: Initial Inventory).

Decision Variables:
- \( x_i \): Number of units of product \( i \in I \) to fulfill (continuous or integer, as appropriate; \( x_i \geq 0 \)).

Objective:
\[
\text{Maximize} \quad Z = \sum_{i \in I} r_i x_i
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
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in EuropeSalesRecords.csv (table_id: file_0_view_0) where Product Name has prefix 'Baby'.
- Parameter \( r_i \): EuropeSalesRecords.csv, column: Revenue, table_id: file_0_view_0.
- Parameter \( d_i \): EuropeSalesRecords.csv, column: Demand, table_id: file_0_view_0.
- Parameter \( s_i \): EuropeSalesRecords.csv, column: Initial Inventory, table_id: file_0_view_0.