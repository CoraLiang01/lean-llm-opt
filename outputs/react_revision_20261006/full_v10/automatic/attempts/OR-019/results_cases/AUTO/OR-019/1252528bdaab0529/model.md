Mathematical Model

Index Sets:
Let $I$ be the set of all products with "Product Name" beginning with "27in" in table_id file_0_view_0.

Parameters:
For each $i \in I$:
$A_i$ = Revenue per unit of product $i$ (from column "Revenue", file_0_view_0)
$d_i$ = Demand for product $i$ (from column "Demand", file_0_view_0)
$I_i$ = Initial Inventory of product $i$ (from column "Initial Inventory", file_0_view_0)

Decision Variables:
For each $i \in I$:
$x_i$ = Number of units of product $i$ to fulfill (integer, $0 \leq x_i \leq \min\{d_i, I_i\}$)

Objective:
$\max \sum_{i \in I} A_i x_i$

Constraints:
For all $i \in I$:
$x_i \leq d_i$
$x_i \leq I_i$
$x_i \in \mathbb{Z}_{\geq 0}$

Data Mapping:
All parameters are mapped from table_id file_0_view_0 in SalesDataAnalysis.csv:
- Index set $I$: all rows where "Product Name" starts with "27in"
- $A_i$: column "Revenue"
- $d_i$: column "Demand"
- $I_i$: column "Initial Inventory"