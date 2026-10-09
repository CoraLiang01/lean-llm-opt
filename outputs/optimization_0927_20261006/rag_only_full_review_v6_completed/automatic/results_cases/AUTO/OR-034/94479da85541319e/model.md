Abstract Optimization Model

Index Sets:
- Let \( I \) be the set of all baked goods, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of baked good \( i \) (from column 'Revenue').
- \( s_i \): Initial inventory of baked good \( i \) (from column 'Initial Inventory').
- \( d_i \): Demand for baked good \( i \) (from column 'Demand').

Decision Variables:
- \( x_i \geq 0 \): Quantity of baked good \( i \) to fulfill (continuous or integer, as appropriate).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales.)

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id 'file_0_view_0', column 'Product Name'.
- Parameter \( r_i \): table_id 'file_0_view_0', column 'Revenue'.
- Parameter \( s_i \): table_id 'file_0_view_0', column 'Initial Inventory'.
- Parameter \( d_i \): table_id 'file_0_view_0', column 'Demand'.

No filters were applied; all records are included as per the original query.