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
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0.
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0.
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0.
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0.
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from table_id file_2_view_0, with row axis "Warehouse (i)" (file_0_view_0) and column axis "Store (j)" (file_1_view_0).

##### Data Mapping

- Warehouses $I$: file_0_view_0, column "Warehouse (i)"
- Stores $J$: file_1_view_0, column "Store (j)"
- Opening cost $f_i$: file_0_view_0, column "Opening Cost (fi)"
- Capacity $u_i$: file_0_view_0, column "Capacity (units)"
- Demand $d_j$: file_1_view_0, column "Demand (units, dj)"
- Transportation cost $c_{ij}$: file_2_view_0, matrix with row axis "Warehouse (i)" (file_0_view_0), column axis "Store (j)" (file_1_view_0)