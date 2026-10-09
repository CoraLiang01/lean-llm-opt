Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘Baby’, indexed by \( i \).
  (Data Mapping: All records in Salesdata.csv where Product Name has prefix "Baby"; table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data Mapping: Salesdata.csv, table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \).
  (Data Mapping: Salesdata.csv, table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \).
  (Data Mapping: Salesdata.csv, table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer)}
\end{align*}
\]

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from Salesdata.csv (table_id: file_0_view_0), using only records where Product Name has prefix "Baby". Columns used: Product Name, Revenue, Demand, Initial Inventory. No other filters applied.