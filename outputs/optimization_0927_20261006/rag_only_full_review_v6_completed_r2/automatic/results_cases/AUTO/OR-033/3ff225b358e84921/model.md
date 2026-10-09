Sets and Indices:
- Let \( I \) be the set of all products classified under ‘Baby’, indexed by \( i \in I \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column Revenue, table_id: file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from column Initial Inventory, table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column Demand, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer if required)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id: file_0_view_0, where "Product Name" has prefix "Baby" (CSVQA-applied filter).
- \( r_i \): "Revenue" column, table_id: file_0_view_0.
- \( s_i \): "Initial Inventory" column, table_id: file_0_view_0.
- \( d_i \): "Demand" column, table_id: file_0_view_0.