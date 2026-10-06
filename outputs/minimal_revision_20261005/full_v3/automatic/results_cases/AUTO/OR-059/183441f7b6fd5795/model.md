##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i$ to dealership $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise, for all $i \in I$.

##### Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id file_1_view_0 and file_2_view_0.
- $J$: Set of dealerships, from column "customer" in table_id file_0_view_0 and columns "C1"–"C9" in file_2_view_0.
- $d_j$: Demand of dealership $j$, from column "demand" in table_id file_0_view_0.
- $f_i$: Fixed cost to open supplier $i$, from column "fixed_costs" in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership Demand Satisfaction**  
   For each dealership $j \in J$:
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier Activation**  
   For each supplier $i \in I$:
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all dealerships).

3. **Variable Domains**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$: supplier set from file_1_view_0["Unnamed: 0"] and file_2_view_0["Unnamed: 0"]
- $J$: dealership set from file_0_view_0["customer"] and file_2_view_0 columns ["C1", ..., "C9"]
- $d_j$: file_0_view_0, columns: "customer", "demand"
- $f_i$: file_1_view_0, columns: "Unnamed: 0", "fixed_costs"
- $c_{ij}$: file_2_view_0, rows: "Unnamed: 0" (supplier), columns: "C1"–"C9" (dealership)
- $M = \sum_{j \in J} d_j$, with $d_j$ from file_0_view_0["demand"]

No additional capacity or supply constraints are imposed beyond those above. All parameters are mapped directly to the supplied CSV data.