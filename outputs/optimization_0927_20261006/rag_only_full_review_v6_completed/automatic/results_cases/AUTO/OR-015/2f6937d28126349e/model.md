Abstract Optimization Model

Index Sets:
- \( I \): Set of ‘Aalop’ products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from [Revenue]).
- \( d_i \): Demand for product \( i \) during the sales horizon (from [Demand]).
- \( s_i \): Initial inventory of product \( i \) (from [Initial Inventory]).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(integer units)}
\end{align*}
\]

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from table_id: file_0_view_0, columns: 
  - [Product Name] (prefix = 'Aalop') → defines \( I \)
  - [Revenue] → \( r_i \)
  - [Demand] → \( d_i \)
  - [Initial Inventory] → \( s_i \)
- The filter [Product Name] prefix = 'Aalop' is applied exactly as validated by CSVQA.