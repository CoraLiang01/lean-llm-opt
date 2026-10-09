##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to customer $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether warehouse $i$ is opened

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Warehouse capacity: $\sum_{j \in J} x_{ij} \leq u_i,\quad \forall i \in I$
3. Activation: $x_{ij} \leq d_j y_i,\quad \forall i \in I,\, j \in J$
4. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

Where:  
$I$ = set of warehouse IDs from file_1_view_0["Warehouse ID"]  
$J$ = set of customer IDs from file_2_view_0["Customer ID"]  
$c_{ij}$ = transportation cost from warehouse $i$ to customer $j$ from file_0_view_0, row "Warehouse ID" $i$, column $j$  
$f_i$ = fixed cost for warehouse $i$ from file_1_view_0["Fixed_Cost"]  
$u_i$ = capacity of warehouse $i$ from file_1_view_0["Capacity"]  
$d_j$ = demand of customer $j$ from file_2_view_0["Demand"]

##### Data Mapping

- $I$: file_1_view_0["Warehouse ID"]
- $J$: file_2_view_0["Customer ID"]
- $c_{ij}$: file_0_view_0, row "Warehouse ID" $i$, column $j$
- $f_i$: file_1_view_0["Fixed_Cost"]
- $u_i$: file_1_view_0["Capacity"]
- $d_j$: file_2_view_0["Demand"]