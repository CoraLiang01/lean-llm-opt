Sets and Indices:
- Let \( I \) be the set of all products where ‘Product Name’ starts with "27in" (from Salesorders.csv).

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (‘Revenue’ column, table_id: file_0_view_0)
- \( s_i \): Initial inventory of product \( i \) (‘Initial Inventory’ column, table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) (‘Demand’ column, table_id: file_0_view_0)

Decision Variables:
- \( x_i \geq 0 \): Number of units of product \( i \) to fulfill (continuous or integer, as appropriate)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)}
\end{align*}
\]

Data Mapping:
- Set \( I \): All records in Salesorders.csv (table_id: file_0_view_0) where ‘Product Name’ has prefix "27in" (CSVQA-applied filter: Product Name prefix "27in").
- Parameter \( r_i \): ‘Revenue’ column, table_id: file_0_view_0.
- Parameter \( s_i \): ‘Initial Inventory’ column, table_id: file_0_view_0.
- Parameter \( d_i \): ‘Demand’ column, table_id: file_0_view_0.