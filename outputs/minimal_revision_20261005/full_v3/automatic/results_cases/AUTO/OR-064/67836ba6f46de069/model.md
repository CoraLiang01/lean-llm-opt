##### Decision Variables

- $x_{ij} \geq 0$: quantity of goods shipped from supplier $i$ to supermarket $j$, for all suppliers $i$ and supermarkets $j$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Parameters

- $f_i$: fixed cost for opening supplier $i$.
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$.
- $d_j$: demand of supermarket $j$.
- $M = \sum_{j} d_j$: a sufficiently large constant (total demand), used to enforce that inactive suppliers cannot ship goods.

##### Sets

- $I$: set of suppliers (from fixed_cost.csv, table_id: file_1_view_0, column: Unnamed: 0)
- $J$: set of supermarkets (from demand.csv, table_id: file_0_view_0, column: customer)

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation:**  
   For each supplier $i \in I$,
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

- $I$: supplier set from table_id: file_1_view_0, column: Unnamed: 0
- $J$: supermarket set from table_id: file_0_view_0, column: customer
- $f_i$: fixed_costs from table_id: file_1_view_0, column: fixed_costs, indexed by $i$
- $d_j$: demand from table_id: file_0_view_0, column: demand, indexed by $j$
- $c_{ij}$: transportation cost matrix from table_id: file_2_view_0, row_id_mapping $i$ (from Unnamed: 0), column_id_mapping $j$ (from header)
- $M = \sum_{j \in J} d_j$ (sum over all $d_j$ from demand.csv)

##### Notes

- The transportation cost matrix aligns suppliers (rows) and supermarkets (columns) using the provided row_id_mapping and column_id_mapping.
- All parameters are directly mapped to the CSV data as described above.