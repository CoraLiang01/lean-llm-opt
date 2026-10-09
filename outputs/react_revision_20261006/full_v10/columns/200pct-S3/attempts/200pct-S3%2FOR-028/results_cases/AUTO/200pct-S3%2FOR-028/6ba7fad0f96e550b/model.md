##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction:  
$\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. Warehouse capacity:  
$\sum_{j \in J} x_{ij} \leq u_i y_i \quad \forall i \in I$

3. Variable domains:  
$x_{ij} \geq 0$ continuous  
$y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of warehouses, from PotentialWarehouses_Costs.csv, column "Warehouse (i)", table_id: file_0_view_0
- $J$: set of stores, from Stores_Demands.csv, column "Store (j)", table_id: file_1_view_0
- $f_i$: opening cost of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Opening Cost (fi)", table_id: file_0_view_0
- $u_i$: capacity of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Capacity (units)", table_id: file_0_view_0
- $d_j$: demand of store $j$, from Stores_Demands.csv, column "Demand (units, dj)", table_id: file_1_view_0
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, table_id: file_2_view_0, with warehouse $i$ as both row and column identifiers ("Unnamed: 3" and "W1", ..., "W11")

All index sets and parameters are defined by the full set of entities in the respective columns of the source files.