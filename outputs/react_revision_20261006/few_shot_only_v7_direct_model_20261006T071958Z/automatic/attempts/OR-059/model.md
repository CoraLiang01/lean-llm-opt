##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: transportation cost per vehicle from supplier $i$ to dealership $j$
- $f_i$: fixed cost to open supplier $i$

##### Constraints

1. **Dealership Demand Satisfaction**  
   For each dealership $j \in J$:
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand of dealership $j$.

2. **Supplier Activation Constraint**  
   For each supplier $i \in I$:
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i
   \]
   where $M_i$ is a sufficiently large upper bound (e.g., $M_i = \sum_{j \in J} d_j$), ensuring that if $y_i = 0$, then $x_{ij} = 0$ for all $j$.

3. **Variable Domains**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Index Sets and Parameter Mapping

- $I$: Set of suppliers, from column `"Unnamed: 0"` in table_id `"file_1_view_0"` (`fixed_cost.csv`)
- $J$: Set of dealerships, from column `"customer"` in table_id `"file_0_view_0"` (`demand.csv`)
- $d_j$: Demand for dealership $j$, from column `"demand"` in table_id `"file_0_view_0"` (`demand.csv`)
- $f_i$: Fixed cost for supplier $i$, from column `"fixed_costs"` in table_id `"file_1_view_0"` (`fixed_cost.csv`)
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from column $j$ in table_id `"file_2_view_0"` (`transportation_costs.csv`), with supplier $i$ identified by `"Unnamed: 0"`
- $M_i$: For all $i$, $M_i = \sum_{j \in J} d_j$ (sum of all dealership demands, computed from `"demand"` in `"file_0_view_0"`)

##### Data Mapping

- Supplier set $I$: `"file_1_view_0"`, column `"Unnamed: 0"`
- Dealership set $J$: `"file_0_view_0"`, column `"customer"`
- Demand $d_j$: `"file_0_view_0"`, columns `"customer"`, `"demand"`
- Fixed cost $f_i$: `"file_1_view_0"`, columns `"Unnamed: 0"`, `"fixed_costs"`
- Transportation cost $c_{ij}$: `"file_2_view_0"`, row `"Unnamed: 0" = i"`, column $j$
- $M_i$: $\sum_{j \in J} d_j$ from `"file_0_view_0"`, column `"demand"`

All parameters and sets are defined directly from the supplied CSV data. No additional constraints or bounds are imposed beyond those specified above.