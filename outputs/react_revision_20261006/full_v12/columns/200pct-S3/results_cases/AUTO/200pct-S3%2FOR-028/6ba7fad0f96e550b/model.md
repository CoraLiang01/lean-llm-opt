##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise, for all $i \in I$.
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

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
   y_i \in \{0,1\}, \quad x_{ij} \geq 0, \quad \forall i \in I, j \in J
   \]

##### Index Sets and Parameters

- $I$: set of warehouses, from column "Warehouse (i)" in table_id file_0_view_0.
- $J$: set of stores, from column "Store (j)" in table_id file_1_view_0.
- $f_i$: opening cost of warehouse $i$, from column "Opening Cost (fi)" in table_id file_0_view_0.
- $u_i$: capacity of warehouse $i$, from column "Capacity (units)" in table_id file_0_view_0.
- $d_j$: demand of store $j$, from column "Demand (units, dj)" in table_id file_1_view_0.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$, from table_id file_2_view_0, with warehouse $i$ as column and store $j$ as row.

##### Data Mapping

- $I$: all values in "Warehouse (i)" (file_0_view_0)
- $J$: all values in "Store (j)" (file_1_view_0)
- $f_i$: "Opening Cost (fi)" (file_0_view_0), keyed by "Warehouse (i)"
- $u_i$: "Capacity (units)" (file_0_view_0), keyed by "Warehouse (i)"
- $d_j$: "Demand (units, dj)" (file_1_view_0), keyed by "Store (j)"
- $c_{ij}$: value at [row $j$ = "Store (j)" in file_1_view_0, column $i$ = "Warehouse (i)" in file_0_view_0] in file_2_view_0

All index sets and parameters are defined by the full set of records in their respective columns.