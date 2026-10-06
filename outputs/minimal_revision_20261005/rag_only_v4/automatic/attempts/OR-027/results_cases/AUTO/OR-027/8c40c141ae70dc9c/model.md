Mathematical Optimization Model

Sets:
- \( I \): Set of products, indexed by \( i \), where each \( i \) corresponds to a unique value in file_0_view_0['Sub Category'].

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (file_0_view_0['Revenue']).
- \( d_i \): Demand quantity for product \( i \) (file_0_view_0['Demand']).
- \( s_i \): Initial inventory of product \( i \) (file_0_view_0['Initial Inventory']).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Demand fulfillment and inventory limits:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping

- Set \( I \): All unique values in file_0_view_0['Sub Category']
- Parameter \( r_i \): file_0_view_0['Revenue'] (indexed by 'Sub Category')
- Parameter \( d_i \): file_0_view_0['Demand'] (indexed by 'Sub Category')
- Parameter \( s_i \): file_0_view_0['Initial Inventory'] (indexed by 'Sub Category')