##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous)  
$y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Store demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. Warehouse capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i,\quad \forall i \in I$

3. Linking warehouse opening and shipments:  
   $x_{ij} \leq u_i y_i,\quad \forall i \in I,\, \forall j \in J$

4. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0 (PotentialWarehouses_Costs.csv)
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0 (Stores_Demands.csv)
- $f_i$: opening cost for warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from table_id file_2_view_0 (TransportationCost.csv), with warehouse $i$ as column and store $j$ as row, using the relationships mapping in the Observation

All index sets, parameters, and matrix axes are defined exactly as in the source data.