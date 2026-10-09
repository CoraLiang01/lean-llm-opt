Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Source: OnlineSalesDataset.csv, column 'Revenue')
- \( s_i \): Initial inventory available for product \( i \). (Source: OnlineSalesDataset.csv, column 'Initial Inventory')
- \( d_i \): Deterministic demand for product \( i \) over the sales horizon. (Source: OnlineSalesDataset.csv, column 'Demand')

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill for customer purchases. (\( x_i \geq 0 \), integer or continuous as appropriate)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from all products.)

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than available inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(Non-negativity)}
\end{align*}
\]

Data Mapping:
- Index set \( I \) is defined by all records in OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Product Name'.
- Parameter \( r_i \) is mapped from OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Revenue'.
- Parameter \( s_i \) is mapped from OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Initial Inventory'.
- Parameter \( d_i \) is mapped from OnlineSalesDataset.csv (table_id: file_0_view_0), column 'Demand'.
- All 119 records are used directly, as returned by the query (no filters applied).

This model maximizes total revenue by optimally allocating inventory to meet deterministic demand, subject to inventory and demand limits for each product.