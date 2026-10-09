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

- $I$: set of warehouses (from column "Warehouse (i)" in PotentialWarehouses_Costs.csv and "Unnamed: 1" in TransportationCost.csv, e.g., $I = \{\text{W1}, \ldots, \text{W11}\}$)
- $J$: set of stores (from column "Store (j)" in Stores_Demands.csv and columns "W1" to "W11" in TransportationCost.csv, $J = \{\text{W1}, \ldots, \text{W11}\}$)
- $f_i$: opening cost of warehouse $i$ (column "Opening Cost (fi)", table_id: file_0_view_0)
- $u_i$: capacity of warehouse $i$ (column "Capacity (units)", table_id: file_0_view_0)
- $d_j$: demand of store $j$ (column "Demand (units, dj)", table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (entry at row $i$ ["Unnamed: 1"], column $j$ in table_id: file_2_view_0)

##### Data Mapping

- Warehouses $I$ and their parameters $f_i$, $u_i$:  
  - Table: PotentialWarehouses_Costs.csv (table_id: file_0_view_0), columns "Warehouse (i)", "Opening Cost (fi)", "Capacity (units)"
- Stores $J$ and their demands $d_j$:  
  - Table: Stores_Demands.csv (table_id: file_1_view_0), columns "Store (j)", "Demand (units, dj)"
- Transportation costs $c_{ij}$:  
  - Table: TransportationCost.csv (table_id: file_2_view_0), matrix with row index "Unnamed: 1" (warehouse $i$), column headers "W1"..."W11" (store $j$)

All index sets, parameters, and constraints are defined directly from the current CSV data as described above.