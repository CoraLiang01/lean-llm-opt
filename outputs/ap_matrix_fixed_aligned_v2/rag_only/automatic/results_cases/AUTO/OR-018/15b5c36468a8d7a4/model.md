Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified under ‘Baby’, as identified by rows in Salesdata.csv where "Product Name" has prefix "Baby".

Parameters:
- \( r_i \): Revenue per unit for product \( i \in I \) (from Salesdata.csv, column "Revenue", table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \in I \) (from Salesdata.csv, column "Demand", table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \in I \) (from Salesdata.csv, column "Initial Inventory", table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \in I \) to fulfill (integer, \( x_i \geq 0 \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of ‘Baby’ products.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer if required)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in Salesdata.csv (table_id: file_0_view_0) where "Product Name" has prefix "Baby" (CSVQA-applied filter: {"column": "Product Name", "operator": "prefix", "value": "Baby"}).
- \( r_i \): "Revenue" column, table_id: file_0_view_0.
- \( d_i \): "Demand" column, table_id: file_0_view_0.
- \( s_i \): "Initial Inventory" column, table_id: file_0_view_0.

No restocking is allowed, and all constraints and data are applied exactly as specified in the query and the CSVQA result.