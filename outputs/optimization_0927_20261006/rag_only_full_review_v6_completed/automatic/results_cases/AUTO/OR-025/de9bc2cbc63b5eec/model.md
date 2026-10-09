Abstract Mathematical Optimization Model

Index Sets:
- Let \( I \) be the set of all smartphone models classified as ‘TABLET’, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit for model \( i \) (from column 'Revenue').
- \( s_i \): Initial inventory for model \( i \) (from column 'Initial Inventory').
- \( d_i \): Demand for model \( i \) (from column 'Demand').

Decision Variables:
- \( x_i \geq 0 \): Number of units of model \( i \) to fulfill (integer, \( x_i \in \mathbb{Z}_+ \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer)}
\end{align*}
\]

Data Mapping:
- Index set \( I \), parameters \( r_i \), \( s_i \), and \( d_i \) are sourced from SmartphoneRetailOutletSalesData.csv (table_id: file_0_view_0), using only records where the 'Product Name' column has prefix 'TABLET_' (i.e., WHERE Product Name LIKE 'TABLET_%').
- \( r_i \) maps to column 'Revenue'.
- \( s_i \) maps to column 'Initial Inventory'.
- \( d_i \) maps to column 'Demand'.

No additional constraints or requirements are imposed beyond those stated above.