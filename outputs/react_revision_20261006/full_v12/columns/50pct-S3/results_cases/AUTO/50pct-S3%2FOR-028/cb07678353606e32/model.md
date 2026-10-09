##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for each warehouse $i$ in $I$ (from PotentialWarehouses_Costs.csv, column "Warehouse (i)").
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for each $i \in I$, $j \in J$ (from PotentialWarehouses_Costs.csv and Stores_Demands.csv).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where:
- $f_i$ is the opening cost of warehouse $i$ (PotentialWarehouses_Costs.csv, "Opening Cost (fi)"),
- $c_{ij}$ is the transportation cost per unit from warehouse $i$ to store $j$ (TransportationCost.csv, entry for warehouse $i$ and store $j$).

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$ (Stores_Demands.csv, "Store (j)"):
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand of store $j$ (Stores_Demands.csv, "Demand (units, dj)").

2. **Warehouse capacity:**  
   For each warehouse $i \in I$:
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i
   \]
   where $K_i$ is the capacity of warehouse $i$ (PotentialWarehouses_Costs.csv, "Capacity (units)").

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets

- $I$: set of warehouses, from PotentialWarehouses_Costs.csv, "Warehouse (i)".
- $J$: set of stores, from Stores_Demands.csv, "Store (j)".

##### Data Mapping

- $f_i$: PotentialWarehouses_Costs.csv, table_id: file_0_view_0, column: "Opening Cost (fi)", index: "Warehouse (i)"
- $K_i$: PotentialWarehouses_Costs.csv, table_id: file_0_view_0, column: "Capacity (units)", index: "Warehouse (i)"
- $d_j$: Stores_Demands.csv, table_id: file_1_view_0, column: "Demand (units, dj)", index: "Store (j)"
- $c_{ij}$: TransportationCost.csv, table_id: file_2_view_0, row: "Unnamed: 1" (warehouse label), column: $Wk$ (store label, $k$ matches "Store (j)" in Stores_Demands.csv)

All index sets and parameters are defined by the full set of unique IDs in the respective columns of the source files.