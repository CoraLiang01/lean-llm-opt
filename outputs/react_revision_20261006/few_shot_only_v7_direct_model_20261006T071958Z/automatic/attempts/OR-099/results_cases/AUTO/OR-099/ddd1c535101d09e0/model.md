##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise.

##### Parameters

- $f_i$: opening cost for warehouse $i$.
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$.
- $u_i$: capacity of warehouse $i$.
- $d_j$: demand of store $j$.

##### Index Sets

- $I$: set of warehouses, from column "Warehouse (i)" in table_id="file_0_view_0".
- $J$: set of stores, from column "Store (j)" in table_id="file_1_view_0".

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

##### Data Mapping

- $I$: All values in column "Warehouse (i)" of table_id="file_0_view_0"
- $f_i$: "Opening Cost (fi)" in table_id="file_0_view_0", indexed by "Warehouse (i)"
- $u_i$: "Capacity (units)" in table_id="file_0_view_0", indexed by "Warehouse (i)"
- $J$: All values in column "Store (j)" of table_id="file_1_view_0"
- $d_j$: "Demand (units, dj)" in table_id="file_1_view_0", indexed by "Store (j)"
- $c_{ij}$: Entry in table_id="file_2_view_0", row with "Unnamed: 0" = "W$i$", column "W$j$", where $i$ and $j$ correspond to warehouse and store indices as per $I$ and $J$.

All index sets and parameters are defined exactly as in the supplied data. No values are enumerated here; see the Observation for all coefficients.