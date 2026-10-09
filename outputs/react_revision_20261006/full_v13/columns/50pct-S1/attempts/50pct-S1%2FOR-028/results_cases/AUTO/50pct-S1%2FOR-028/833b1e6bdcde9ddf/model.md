##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all warehouses $i$.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for all warehouses $i$ and stores $j$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand is fully met.)

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
   (A warehouse cannot ship more than its capacity, and only if it is open.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from column "Warehouse (i)" in table_id: file_0_view_0 (PotentialWarehouses_Costs.csv).
- $J$: Set of stores, from column "Store (j)" in table_id: file_1_view_0 (Stores_Demands.csv).
- $f_i$: Opening cost for warehouse $i$, from column "Opening Cost (fi)" in table_id: file_0_view_0.
- $u_i$: Capacity of warehouse $i$, from column "Capacity (units)" in table_id: file_0_view_0.
- $d_j$: Demand of store $j$, from column "Demand (units, dj)" in table_id: file_1_view_0.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from table_id: file_2_view_0 (TransportationCost.csv), with warehouse and store mapping as follows:
    - Warehouse $i$ corresponds to "W" + value of "Warehouse (i)" in file_0_view_0.
    - Store $j$ corresponds to row with "Unnamed: 1" = "W" + value of "Store (j)" in file_2_view_0; the cost is in column "W" + value of "Warehouse (i)".

##### Data Mapping

- Warehouses $I$: file_0_view_0, column "Warehouse (i)"
- Warehouse opening cost $f_i$: file_0_view_0, column "Opening Cost (fi)"
- Warehouse capacity $u_i$: file_0_view_0, column "Capacity (units)"
- Stores $J$: file_1_view_0, column "Store (j)"
- Store demand $d_j$: file_1_view_0, column "Demand (units, dj)"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 1" = "W" + $j$, column "W" + $i$

All indices and parameters are to be taken directly from the referenced columns and rows in the source tables.