##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. Warehouse capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i \quad \forall i \in I$

3. Activation logic:  
   $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$  
   $y_i \in \{0,1\} \quad \forall i \in I$

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry at row $i$, column $j$ in table_id file_2_view_0, with warehouse and store IDs mapped as per the relationships in the Observation

##### Data Mapping

- Warehouses $I$ and their parameters $f_i$, $u_i$: from table_id file_0_view_0, columns "Warehouse (i)", "Opening Cost (fi)", "Capacity (units)"
- Stores $J$ and their demands $d_j$: from table_id file_1_view_0, columns "Store (j)", "Demand (units, dj)"
- Transportation costs $c_{ij}$: from table_id file_2_view_0, with warehouse and store IDs mapped according to the relationships in the Observation (matrix: rows and columns correspond to warehouses and stores as per the mapping)
- All indices and parameters are to be taken exactly as present in the source files; no abbreviation or simplification.