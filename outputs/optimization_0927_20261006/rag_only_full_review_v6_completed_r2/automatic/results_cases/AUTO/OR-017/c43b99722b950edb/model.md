Abstract Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘ZZ’, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \) (from [Revenue]).
- \( s_i \): Initial inventory for product \( i \) (from [Initial Inventory]).
- \( d_i \): Demand for product \( i \) (from [Demand]).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot exceed initial inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot exceed demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(integer units)}
\end{align*}
\]

Data Mapping:
- Table: RetailStoreSalesTransactions(ScannerData).csv (table_id: file_0_view_0)
- Index set \( I \): All records where [SKU] has prefix 'ZZ' (i.e., [SKU] = 'ZZ*')
- Parameters:
    - \( r_i \): [Revenue] column, for each \( i \in I \)
    - \( s_i \): [Initial Inventory] column, for each \( i \in I \)
    - \( d_i \): [Demand] column, for each \( i \in I \)
- The subset is defined by the filter: [SKU] has prefix 'ZZ' (as applied in the query; do not re-filter).

This model maximizes total revenue from fulfilling demand for products in category ‘ZZ’, subject to inventory and demand limits for each product.