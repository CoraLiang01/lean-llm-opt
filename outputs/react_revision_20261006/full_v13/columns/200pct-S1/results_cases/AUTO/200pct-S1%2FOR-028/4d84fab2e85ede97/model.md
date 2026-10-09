##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand must be fully met.)

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
   (A warehouse cannot ship more than its capacity, and only if it is open.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $J$: set of stores, from column "Store (j)" in Stores_Demands.csv (table_id: file_1_view_0)
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in PotentialWarehouses_Costs.csv (table_id: file_0_view_0)
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in Stores_Demands.csv (table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from entry at row with "Unnamed: 3" = $W_i$ and column $W_j$ in TransportationCost.csv (table_id: file_2_view_0, with $i$ and $j$ mapped as $W1,\ldots,W11$)

##### Data Mapping

- Warehouses $I$: file_0_view_0, column "Warehouse (i)"
- Warehouse opening cost $f_i$: file_0_view_0, column "Opening Cost (fi)"
- Warehouse capacity $u_i$: file_0_view_0, column "Capacity (units)"
- Stores $J$: file_1_view_0, column "Store (j)"
- Store demand $d_j$: file_1_view_0, column "Demand (units, dj)"
- Transportation cost $c_{ij}$: file_2_view_0, row where "Unnamed: 3" = $W_i$, column $W_j$

All index sets and parameters are defined exactly as in the source data. No values are enumerated here; see the referenced columns and table_ids for all coefficients.