Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products classified under ‘FAUX’. (Each product \( i \in I \) corresponds to a row returned by the query.)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (From column Revenue, table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \). (From column Demand, table_id: file_0_view_0)
- \( s_i \): Initial inventory for product \( i \). (From column Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& 0 \leq x_i \leq d_i && \forall i \in I \\
& 0 \leq x_i \leq s_i && \forall i \in I \\
& x_i \in \mathbb{Z}_+ && \forall i \in I
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id: file_0_view_0 (ZARASales.csv) where Product Name has prefix 'FAUX'.
- \( r_i \): Revenue, table_id: file_0_view_0, column: Revenue
- \( d_i \): Demand, table_id: file_0_view_0, column: Demand
- \( s_i \): Initial Inventory, table_id: file_0_view_0, column: Initial Inventory

All data and constraints are based solely on the records and columns returned by the validated query. No additional filtering or assumptions have been made.