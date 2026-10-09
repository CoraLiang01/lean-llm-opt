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
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of warehouses, from column `"Warehouse (i)"` in table_id `file_0_view_0` of PotentialWarehouses_Costs.csv.
- $J$: Set of stores, from column `"Store (j)"` in table_id `file_1_view_0` of Stores_Demands.csv.
- $f_i$: Opening cost for warehouse $i$, from column `"Opening Cost (fi)"` in table_id `file_0_view_0`.
- $u_i$: Capacity of warehouse $i$, from column `"Capacity (units)"` in table_id `file_0_view_0`.
- $d_j$: Demand of store $j$, from column `"Demand (units, dj)"` in table_id `file_1_view_0`.
- $c_{ij}$: Transportation cost from warehouse $i$ to store $j$, from table_id `file_2_view_0` of TransportationCost.csv, with warehouse $i$ corresponding to row `"Unnamed: 1"` and store $j$ corresponding to column `"Wk"` (where $k$ matches warehouse/store indices as per the file's mapping).

##### Data Mapping

- Warehouses $I$: `"Warehouse (i)"`, `file_0_view_0`
- Stores $J$: `"Store (j)"`, `file_1_view_0`
- Opening cost $f_i$: `"Opening Cost (fi)"`, `file_0_view_0`
- Capacity $u_i$: `"Capacity (units)"`, `file_0_view_0`
- Demand $d_j$: `"Demand (units, dj)"`, `file_1_view_0`
- Transportation cost $c_{ij}$: `"Wk"` columns, `file_2_view_0`, with warehouse $i$ as `"Unnamed: 1"` row and store $j$ as column

No values are enumerated; all parameters are mapped directly to their CSV sources.