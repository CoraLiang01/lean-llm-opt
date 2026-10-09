##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Supermarket demand:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Domains:  
   $x_{ij} \geq 0$ (continuous), $\quad y_i \in \{0,1\}$

3. (Optional, if required by logic) Linking:  
   $x_{ij} \leq d_j y_i, \quad \forall i \in I, \forall j \in J$  
   (This ensures that if supplier $i$ is not open, it cannot supply any supermarket.)

##### Index Sets and Data Mapping

- $I$: set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" (fixed_cost.csv)
- $J$: set of supermarkets, from column "customer" in table_id "file_0_view_0" (demand.csv)
- $d_j$: demand of supermarket $j$, from column "demand" in table_id "file_0_view_0" (demand.csv)
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0" (fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from table_id "file_2_view_0" (transportation_costs.csv), with rows indexed by "Unnamed: 0" (supplier) and columns by supermarket IDs ("C1", ..., "C25")

All parameters and sets are mapped directly from the provided CSV files as described above.