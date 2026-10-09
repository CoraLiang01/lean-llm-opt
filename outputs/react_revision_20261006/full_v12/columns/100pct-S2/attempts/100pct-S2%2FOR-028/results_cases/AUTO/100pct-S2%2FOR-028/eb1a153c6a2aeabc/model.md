##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}, \quad \forall i \in I,\, j \in J
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv (table_id: file_1_view_0)
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $K_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv (table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry $(i,j)$ in TransportationCost.csv (table_id: file_2_view_0), with warehouse and store indices mapped as per the relationships in the Observation.

##### Data Mapping

- $I$: all values in "Warehouse (i)" (file_0_view_0)
- $J$: all values in "Store (j)" (file_1_view_0)
- $f_i$: "Opening Cost (fi)" (file_0_view_0), indexed by $i$
- $K_i$: "Capacity (units)" (file_0_view_0), indexed by $i$
- $d_j$: "Demand (units, dj)" (file_1_view_0), indexed by $j$
- $c_{ij}$: entry in TransportationCost.csv (file_2_view_0), row $j$ ("Unnamed: 1"), column $Wk$ (warehouse $i$ as $Wk$), with mapping as per relationships in the Observation

All index sets and parameters are defined by the full set of records in the respective columns of the source files.