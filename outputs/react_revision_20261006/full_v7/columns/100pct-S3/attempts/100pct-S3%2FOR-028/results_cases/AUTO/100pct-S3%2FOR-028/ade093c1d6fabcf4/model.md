##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i \in I$ is opened, 0 otherwise.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).

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
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0.
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0.
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0.
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0.
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from table_id file_2_view_0, with warehouse $i$ as row "Unnamed: 1" and store $j$ as column "Wk" (where $k$ matches warehouse/store indices).

##### Data Mapping

- Warehouses $I$: file_0_view_0, column "Warehouse (i)"
- Warehouse opening cost $f_i$: file_0_view_0, column "Opening Cost (fi)"
- Warehouse capacity $u_i$: file_0_view_0, column "Capacity (units)"
- Stores $J$: file_1_view_0, column "Store (j)"
- Store demand $d_j$: file_1_view_0, column "Demand (units, dj)"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 1" (warehouse $i$), column "Wk" (store $j$)

All index sets and parameters are defined by the full set of records in the respective columns of the current CSV files.