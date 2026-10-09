##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether warehouse $i$ is opened.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from PotentialWarehouses_Costs.csv, column "Warehouse (i)", table_id: file_0_view_0.
- $J$: Set of stores, from Stores_Demands.csv, column "Store (j)", table_id: file_1_view_0.
- $f_i$: Opening cost of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Opening Cost (fi)", table_id: file_0_view_0.
- $u_i$: Capacity of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Capacity (units)", table_id: file_0_view_0.
- $d_j$: Demand of store $j$, from Stores_Demands.csv, column "Demand (units, dj)", table_id: file_1_view_0.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, row with "Unnamed: 1" = $W_i$, column $W_j$, table_id: file_2_view_0.

##### Notes

- $x_{ij}$ is only allowed to be positive if $y_i = 1$ (warehouse $i$ is open).
- All indices and parameters are mapped directly from the provided CSV files as described above.