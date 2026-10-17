Abstract Optimization Model

Index Sets:
- \( I \): Set of all products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Source: SalesDatainBusinesses.csv, column 'Revenue')
- \( d_i \): Demand for product \( i \). (Source: SalesDatainBusinesses.csv, column 'Demand')
- \( s_i \): Initial inventory available for product \( i \). (Source: SalesDatainBusinesses.csv, column 'Initial Inventory')

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (or continuous if partial fulfillment is allowed).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from all fulfilled products.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than available inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Non-negativity)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All rows in SalesDatainBusinesses.csv (table_id: file_0_view_0), column 'Product Name'
- Parameter \( r_i \): SalesDatainBusinesses.csv (table_id: file_0_view_0), column 'Revenue'
- Parameter \( d_i \): SalesDatainBusinesses.csv (table_id: file_0_view_0), column 'Demand'
- Parameter \( s_i \): SalesDatainBusinesses.csv (table_id: file_0_view_0), column 'Initial Inventory'

All products in the table are included; no filters are applied.