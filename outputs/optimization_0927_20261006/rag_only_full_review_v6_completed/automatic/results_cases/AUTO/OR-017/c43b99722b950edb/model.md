ABSTRACT Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘ZZ’, indexed by \( i \in I \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \). [from column Revenue]
- \( s_i \): Initial inventory for product \( i \). [from column Initial Inventory]
- \( d_i \): Demand for product \( i \). [from column Demand]

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min(s_i, d_i) \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of ‘ZZ’ products.)

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min(s_i, d_i) \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand.)

Variable Domains:
- \( x_i \) integer, \( \forall i \in I \).

Data Mapping:
- Source: RetailStoreSalesTransactions(ScannerData).csv, table_id: file_0_view_0
- Index set \( I \): All records where [SKU] has prefix 'ZZ' (CSVQA filter: [SKU] prefix 'ZZ')
- Parameters:
    - \( r_i \): [Revenue] column, for each \( i \in I \)
    - \( s_i \): [Initial Inventory] column, for each \( i \in I \)
    - \( d_i \): [Demand] column, for each \( i \in I \)

No additional constraints or requirements are imposed beyond those stated in the user query and the data returned by CSVQA.