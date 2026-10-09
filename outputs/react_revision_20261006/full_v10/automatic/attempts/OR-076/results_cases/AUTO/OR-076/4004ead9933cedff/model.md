##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. Warehouse capacity:  
   $\sum_{j \in J} x_{ij} \leq u_i,\quad \forall i \in I$

3. Activation linkage:  
   $x_{ij} \leq u_i y_i,\quad \forall i \in I,\, j \in J$

4. Domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: set of warehouses, from column "Warehouse ID" in file_1_view_0 (warehouse.csv)
- $J$: set of customers, from column "Customer ID" in file_2_view_0 (demand.csv)
- $c_{ij}$: unit transportation cost from warehouse $i$ to customer $j$, from file_0_view_0 (cost.csv), row "Warehouse ID" = $i$, column $j$
- $f_i$: fixed opening cost for warehouse $i$, from column "Fixed_Cost" in file_1_view_0 (warehouse.csv)
- $u_i$: capacity of warehouse $i$, from column "Capacity" in file_1_view_0 (warehouse.csv)
- $d_j$: demand of customer $j$, from column "Demand" in file_2_view_0 (demand.csv)

All parameters are mapped directly from the source tables as described above.