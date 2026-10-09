##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for each warehouse $i$.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for each warehouse $i$ and store $j$.

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$

2. **Warehouse capacity:**  
   $\sum_{j \in J} x_{ij} \leq u_i y_i \quad \forall i \in I$

3. **Variable domains:**  
   $y_i \in \{0,1\} \quad \forall i \in I$  
   $x_{ij} \geq 0 \quad \forall i \in I,\, j \in J$

##### Index Sets and Data Mapping

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0 (PotentialWarehouses_Costs.csv)
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0 (Stores_Demands.csv)
- $f_i$: opening cost for warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry at row with "Unnamed: 1" = "W$i$" and column "W$j$" in table_id file_2_view_0 (TransportationCost.csv)

All parameters are mapped directly from the CSV files as described above.