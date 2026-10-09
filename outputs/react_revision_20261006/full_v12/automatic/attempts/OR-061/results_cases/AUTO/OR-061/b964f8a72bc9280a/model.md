##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Branch demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation logic:  
   $\sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I$  
   where $M = \sum_{j \in J} d_j$ (total demand; a valid upper bound since there are no supplier capacity limits).

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" (fixed_cost.csv)
- $J$: set of branches, from column "customer" in table_id "file_0_view_0" (demand.csv)
- $d_j$: demand of branch $j$, from column "demand" in table_id "file_0_view_0"
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0"
- $c_{ij}$: transportation cost per unit from supplier $i$ to branch $j$, from table_id "file_2_view_0" (row: "Unnamed: 0" = $i$, column: $j$)
- $M = \sum_{j \in J} d_j$ (sum over all $d_j$ from demand.csv)

##### Data Mapping

- Suppliers $I$: table_id "file_1_view_0", column "Unnamed: 0"
- Branches $J$: table_id "file_0_view_0", column "customer"
- Demand $d_j$: table_id "file_0_view_0", column "demand"
- Fixed cost $f_i$: table_id "file_1_view_0", column "fixed_costs"
- Transportation cost $c_{ij}$: table_id "file_2_view_0", row "Unnamed: 0" = $i$, column $j$
- $M$: sum of all $d_j$ from table_id "file_0_view_0", column "demand"