##### Symbolic Mathematical Model

Let  
$P$ = set of all products with "27in" in their Product Name from table_id file_0_view_0.

Parameters (for each $i \in P$):  
$A_i$ = Revenue per unit of product $i$ (from column "Revenue")  
$d_i$ = Demand for product $i$ (from column "Demand")  
$I_i$ = Initial Inventory of product $i$ (from column "Initial Inventory")

Decision variables:  
$x_i$ = number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in P$

Objective:  
$\max \sum_{i \in P} A_i \cdot x_i$

Subject to:  
1. Inventory constraints:  
$\quad x_i \leq I_i, \quad \forall i \in P$

2. Demand constraints:  
$\quad x_i \leq d_i, \quad \forall i \in P$

3. Nonnegativity and integrality:  
$\quad x_i \in \mathbb{Z}_+, \quad \forall i \in P$

##### Data Mapping

- Index set $P$ and all parameters $A_i$, $d_i$, $I_i$ are drawn from table_id file_0_view_0, columns:
    - "Product Name" (filtered to names containing "27in")
    - "Revenue"
    - "Demand"
    - "Initial Inventory"