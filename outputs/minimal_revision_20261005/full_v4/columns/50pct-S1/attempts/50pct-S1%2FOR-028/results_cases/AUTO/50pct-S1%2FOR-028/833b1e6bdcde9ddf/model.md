##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $I$: Set of warehouses, from PotentialWarehouses_Costs.csv, column "Warehouse (i)", table_id: file_0_view_0.
- $J$: Set of stores, from Stores_Demands.csv, column "Store (j)", table_id: file_1_view_0.
- $f_i$: Opening cost of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Opening Cost (fi)", table_id: file_0_view_0.
- $u_i$: Capacity of warehouse $i$, from PotentialWarehouses_Costs.csv, column "Capacity (units)", table_id: file_0_view_0.
- $d_j$: Demand of store $j$, from Stores_Demands.csv, column "Demand (units, dj)", table_id: file_1_view_0.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from TransportationCost.csv, entry at row with "Unnamed: 1" = "W$i$" and column "W$j$", table_id: file_2_view_0.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

---

#### Data Mapping

- $I$: All values in PotentialWarehouses_Costs.csv, column "Warehouse (i)", table_id: file_0_view_0
- $J$: All values in Stores_Demands.csv, column "Store (j)", table_id: file_1_view_0
- $f_i$: PotentialWarehouses_Costs.csv, columns "Warehouse (i)", "Opening Cost (fi)", table_id: file_0_view_0
- $u_i$: PotentialWarehouses_Costs.csv, columns "Warehouse (i)", "Capacity (units)", table_id: file_0_view_0
- $d_j$: Stores_Demands.csv, columns "Store (j)", "Demand (units, dj)", table_id: file_1_view_0
- $c_{ij}$: TransportationCost.csv, entry at row with "Unnamed: 1" = "W$i$", column "W$j$", table_id: file_2_view_0

All index sets and parameters are defined by the full set of rows and columns as described above.