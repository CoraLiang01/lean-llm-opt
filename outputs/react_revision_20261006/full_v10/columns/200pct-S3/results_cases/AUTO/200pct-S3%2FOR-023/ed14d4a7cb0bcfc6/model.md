##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand satisfaction:  
   $\sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J$

2. Supplier activation logic:  
   $x_{ij} \leq D_j y_i, \quad \forall i \in I, \forall j \in J$  
   (where $D_j$ is the demand for customer $j$; ensures no shipments from inactive suppliers)

3. Variable domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Data Mapping

- $I$: Set of suppliers, from column "Unnamed: 3" in table_id file_1_view_0 (fixed_cost.csv)
- $J$: Set of customers, from column "Customer" in table_id file_0_view_0 (demand.csv)
- $d_j$: Demand for customer $j$, from column "demand" in table_id file_0_view_0 (demand.csv)
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id file_1_view_0 (fixed_cost.csv)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$, from table_id file_2_view_0 (transportation_costs.csv), with supplier $i$ identified by "Unnamed: 4" and customer $j$ by column names matching store names.
- $D_j$: Demand for customer $j$, as above.

All index sets and parameters are defined by the full set of entities in the respective columns of the current CSV files. No capacity limits are imposed beyond those implied by demand and activation logic.