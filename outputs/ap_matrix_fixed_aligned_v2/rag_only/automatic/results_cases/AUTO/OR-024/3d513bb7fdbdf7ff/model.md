ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products with Product Name starting with 'S700_' (from SampleSalesData.csv, table_id: file_0_view_0).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue', table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column 'Demand', table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) (from column 'Initial Inventory', table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than available inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Non-negativity, integer if required)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in SampleSalesData.csv (table_id: file_0_view_0) where 'Product Name' has prefix 'S700_' (CSVQA-applied filter).
- Parameter \( r_i \): 'Revenue' column, table_id: file_0_view_0.
- Parameter \( d_i \): 'Demand' column, table_id: file_0_view_0.
- Parameter \( s_i \): 'Initial Inventory' column, table_id: file_0_view_0.

No additional constraints or requirements are imposed beyond those specified in the user query and the CSVQA result.