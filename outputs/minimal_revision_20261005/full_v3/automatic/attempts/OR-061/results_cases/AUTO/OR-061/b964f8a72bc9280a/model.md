##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i$ to branch $j$ (continuous), for all suppliers $i$ and branches $j$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $d_j$: Demand of branch $j$.
- $f_i$: Fixed cost to activate supplier $i$.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$.
- $M = \sum_{j} d_j$: A sufficiently large constant (total demand), used to enforce that inactive suppliers cannot ship goods.

##### Sets

- $I$: Set of suppliers (from fixed_cost.csv, table_id: file_1_view_0, column: Unnamed: 0).
- $J$: Set of branches (from demand.csv, table_id: file_0_view_0, column: customer).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each branch $j \in J$,
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

- $I$: supplier identifiers from fixed_cost.csv (table_id: file_1_view_0, column: Unnamed: 0)
- $J$: branch identifiers from demand.csv (table_id: file_0_view_0, column: customer)
- $d_j$: demand for branch $j$ from demand.csv (table_id: file_0_view_0, column: demand)
- $f_i$: fixed cost for supplier $i$ from fixed_cost.csv (table_id: file_1_view_0, column: fixed_costs)
- $c_{ij}$: transportation cost from supplier $i$ to branch $j$ from transportation_costs.csv (table_id: file_2_view_0, row_id_mapping: Unnamed: 0, column_id_mapping: [C1, C2, C3, C4, C5])
- $M = \sum_{j \in J} d_j$: sum of all $d_j$ values from demand.csv (table_id: file_0_view_0, column: demand)