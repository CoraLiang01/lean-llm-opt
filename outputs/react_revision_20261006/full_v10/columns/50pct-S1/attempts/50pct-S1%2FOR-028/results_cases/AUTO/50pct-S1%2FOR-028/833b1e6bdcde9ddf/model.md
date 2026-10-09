##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Warehouse capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i, \quad \forall i \in I$

3. Activation link:  
   $x_{ij} \leq u_i y_i, \quad \forall i \in I, \forall j \in J$

4. Domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- Warehouses $I$:  
  All "Warehouse (i)" from PotentialWarehouses_Costs.csv [table_id: file_0_view_0]

- Stores $J$:  
  All "Store (j)" from Stores_Demands.csv [table_id: file_1_view_0]

- Store demands $d_j$:  
  $d_j$ = "Demand (units, dj)" for store $j$ in Stores_Demands.csv [table_id: file_1_view_0]

- Warehouse opening costs $f_i$:  
  $f_i$ = "Opening Cost (fi)" for warehouse $i$ in PotentialWarehouses_Costs.csv [table_id: file_0_view_0]

- Warehouse capacities $u_i$:  
  $u_i$ = "Capacity (units)" for warehouse $i$ in PotentialWarehouses_Costs.csv [table_id: file_0_view_0]

- Transportation costs $c_{ij}$:  
  $c_{ij}$ = entry in TransportationCost.csv [table_id: file_2_view_0], where row with "Unnamed: 1" = "W$i$" and column "W$j$" gives cost from warehouse $i$ to store $j$.

##### Notes

- All index sets and parameters are defined by the full set of entities in the respective CSV files.
- The activation link ensures that no goods are shipped from unopened warehouses.
- All data mappings refer to the exact columns and table_ids as retrieved.