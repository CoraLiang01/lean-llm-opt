##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. Warehouse capacity:  
   $\sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I$

3. Variable domains:  
   $x_{ij} \geq 0$ continuous  
   $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv (table_id: file_1_view_0)
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $K_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv (table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry (row $j$, column $i$) in TransportationCost.csv (table_id: file_2_view_0, row_id: "Unnamed: 1", column_id: $i$)

##### Data Mapping

- Warehouses $I$: file_0_view_0, column "Warehouse (i)"
- Stores $J$: file_1_view_0, column "Store (j)"
- Opening costs $f_i$: file_0_view_0, column "Opening Cost (fi)"
- Capacities $K_i$: file_0_view_0, column "Capacity (units)"
- Demands $d_j$: file_1_view_0, column "Demand (units, dj)"
- Transportation costs $c_{ij}$: file_2_view_0, row "Unnamed: 1" (store $j$), column $i$ (warehouse $i$)

All index sets and parameters are defined by the full set of records in their respective columns as returned above.