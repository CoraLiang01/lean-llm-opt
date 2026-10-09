Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products with names starting with 'FAUX' (from ZARASales.csv, see Data Mapping).

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue').
- \( s_i \): Initial inventory of product \( i \) (from column 'Initial Inventory').
- \( d_i \): Demand for product \( i \) (from column 'Demand').

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& 0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in I \\
\end{align*}
\]
(Each product’s fulfilled quantity cannot exceed its initial inventory or its demand.)

Variable Domains:
- \( x_i \) are integer variables, \( x_i \geq 0 \).

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( s_i \), \( d_i \) are sourced from ZARASales.csv, table_id: file_0_view_0, columns: 'Product Name' (prefix 'FAUX'), 'Revenue', 'Initial Inventory', 'Demand'. The filter applied is: Product Name starts with 'FAUX'.

No additional constraints or requirements are imposed beyond those specified in the user query and the returned data.