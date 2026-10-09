##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. **Demand satisfaction:**  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. **Warehouse capacity:**  
   $\sum_{j \in J} x_{ij} \leq K_i y_i,\quad \forall i \in I$

3. **Variable domains:**  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from column "Warehouse (i)" in table_id: file_0_view_0 (PotentialWarehouses_Costs.csv)
- $J$: Set of stores, from column "Store (j)" in table_id: file_1_view_0 (Stores_Demands.csv)
- $f_i$: Opening cost for warehouse $i$, from column "Opening Cost (fi)" in table_id: file_0_view_0
- $K_i$: Capacity of warehouse $i$, from column "Capacity (units)" in table_id: file_0_view_0
- $d_j$: Demand of store $j$, from column "Demand (units, dj)" in table_id: file_1_view_0
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from table_id: file_2_view_0 (TransportationCost.csv), with warehouse $i$ as row "Unnamed: 3" and store $j$ as column "Wk" (where $k$ matches warehouse/store indices).

##### Notes on Data Mapping

- Warehouses: $I = \{$values in "Warehouse (i)" from file_0_view_0$\}$
- Stores: $J = \{$values in "Store (j)" from file_1_view_0$\}$
- Transportation cost matrix $c_{ij}$: For warehouse $i$ (row "Unnamed: 3" = "W$i$") and store $j$ (column "W$j$") in file_2_view_0.

All parameters and sets are to be taken exactly as defined in the source tables above.