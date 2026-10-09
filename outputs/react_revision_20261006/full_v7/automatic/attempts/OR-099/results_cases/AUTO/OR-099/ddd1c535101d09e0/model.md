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

##### Index Sets and Data Mapping

- $I$: Set of warehouses, $I = \{$values in column `"Warehouse (i)"` of table_id `file_0_view_0`$\}$
- $J$: Set of stores, $J = \{$values in column `"Store (j)"` of table_id `file_1_view_0`$\}$

##### Parameters and Source Data

- $f_i$: Opening cost of warehouse $i$, from column `"Opening Cost (fi)"` in table_id `file_0_view_0`, indexed by `"Warehouse (i)"`
- $u_i$: Capacity of warehouse $i$, from column `"Capacity (units)"` in table_id `file_0_view_0`, indexed by `"Warehouse (i)"`
- $d_j$: Demand of store $j$, from column `"Demand (units, dj)"` in table_id `file_1_view_0`, indexed by `"Store (j)"`
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$, from table_id `file_2_view_0`, with row index `"Unnamed: 0"` (warehouse, mapped as $i$) and column index $Wj$ (store, mapped as $j$)

##### Data Mapping

- Warehouses $i$: `"Warehouse (i)"` in `file_0_view_0`
- Stores $j$: `"Store (j)"` in `file_1_view_0`
- $f_i$: `"Opening Cost (fi)"` in `file_0_view_0`, indexed by $i$
- $u_i$: `"Capacity (units)"` in `file_0_view_0`, indexed by $i$
- $d_j$: `"Demand (units, dj)"` in `file_1_view_0`, indexed by $j$
- $c_{ij}$: `file_2_view_0`, row `"Unnamed: 0"` = $Wi$, column $Wj$ (store $j$)

No parameters or sets are omitted; all are mapped directly to the CSV source columns and indices.