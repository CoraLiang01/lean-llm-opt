Abstract Optimization Model for Merchandise Allocation

Index Sets:
- \( I \): Set of products/categories, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \). (from Revenue column)
- \( d_i \): Demand quantity for product \( i \). (from Demand column)
- \( s_i \): Initial inventory available for product \( i \). (from Initial Inventory column)

Decision Variables:
- \( x_i \geq 0 \): Quantity of product \( i \) to fulfill (continuous, non-negative).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled quantities.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records returned from RetailSalesDataset.csv, table_id file_0_view_0, filtered where Product Name contains "electronics", "apparel", or "homeware".
- \( r_i \): Revenue column, table_id file_0_view_0.
- \( d_i \): Demand column, table_id file_0_view_0.
- \( s_i \): Initial Inventory column, table_id file_0_view_0.

This model allocates inventory to maximize revenue, subject to demand and inventory constraints, using only the filtered records as returned by CSVQA.