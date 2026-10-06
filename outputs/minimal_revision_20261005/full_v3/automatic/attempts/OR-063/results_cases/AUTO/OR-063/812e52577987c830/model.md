##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods supplied from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I$: Set of warehouses, indexed by $i$ (from table_id: file_1_view_0, column: Unnamed: 0).
- $J$: Set of musicians/bands, indexed by $j$ (from table_id: file_0_view_0, column: customer).
- $d_j$: Demand of musician/band $j$ (from table_id: file_0_view_0, column: demand).
- $f_i$: Fixed cost of activating warehouse $i$ (from table_id: file_1_view_0, column: fixed_costs).
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from table_id: file_2_view_0, row: Unnamed: 0, columns: C1–C7).
- $M$: A sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$ (warehouses): table_id: file_1_view_0, column: Unnamed: 0
- $J$ (musicians/bands): table_id: file_0_view_0, column: customer
- $d_j$: table_id: file_0_view_0, column: demand
- $f_i$: table_id: file_1_view_0, column: fixed_costs
- $c_{ij}$: table_id: file_2_view_0, row_id_mapping: Unnamed: 0 (warehouses), column_id_mapping: C1–C7 (musicians/bands)
- $M = \sum_{j \in J} d_j$ (expression over table_id: file_0_view_0, column: demand)