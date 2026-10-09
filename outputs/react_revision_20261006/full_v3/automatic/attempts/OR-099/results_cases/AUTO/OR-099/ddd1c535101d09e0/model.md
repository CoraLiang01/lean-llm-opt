##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

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
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv and row/column IDs in TransportationCost.csv.
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv and row/column IDs in TransportationCost.csv.
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv.
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv.
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, with warehouse $i$ as row and store $j$ as column.

##### Data Mapping

- $I$: All unique values in "Warehouse (i)" (PotentialWarehouses_Costs.csv, table_id: file_0_view_0) and row/column IDs in TransportationCost.csv (table_id: file_2_view_0).
- $J$: All unique values in "Store (j)" (Stores_Demands.csv, table_id: file_1_view_0) and row/column IDs in TransportationCost.csv (table_id: file_2_view_0).
- $f_i$: "Opening Cost (fi)" for warehouse $i$ (PotentialWarehouses_Costs.csv, table_id: file_0_view_0).
- $u_i$: "Capacity (units)" for warehouse $i$ (PotentialWarehouses_Costs.csv, table_id: file_0_view_0).
- $d_j$: "Demand (units, dj)" for store $j$ (Stores_Demands.csv, table_id: file_1_view_0).
- $c_{ij}$: entry in TransportationCost.csv (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$.

All indices and parameters are defined by the full set of records in the respective columns and tables as described above.