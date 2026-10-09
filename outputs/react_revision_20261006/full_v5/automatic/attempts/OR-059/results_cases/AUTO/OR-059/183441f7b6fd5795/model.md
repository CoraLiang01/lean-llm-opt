##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (total demand).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" and "file_2_view_0".
- $J$: Set of dealerships, from column "customer" in table_id "file_0_view_0" and columns "C1"..."C9" in "file_2_view_0".

##### Parameters and Data Mapping

- $d_j$: Demand of dealership $j$, from column "demand" in table_id "file_0_view_0".
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0".
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from table_id "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M$: $M = \sum_{j \in J} d_j$, with $d_j$ as above.

##### Data Mapping

- $I$: All values in "Unnamed: 0" of "file_1_view_0" and "file_2_view_0".
- $J$: All values in "customer" of "file_0_view_0" and columns "C1"..."C9" of "file_2_view_0".
- $d_j$: "demand" column, table_id "file_0_view_0", key "customer".
- $f_i$: "fixed_costs" column, table_id "file_1_view_0", key "Unnamed: 0".
- $c_{ij}$: Table "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$ as above.

All variables, parameters, and sets are defined directly from the CSV data as described.