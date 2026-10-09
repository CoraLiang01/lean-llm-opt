##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Sets and Indices

- $I$: Set of suppliers, indexed by $i$ (from `file_1_view_0` and `file_2_view_0` rows, e.g., "MOUNT AYR", "WAUKEE", "WAVERLY", "PELLA", "DES MOINES").
- $J$: Set of stores, indexed by $j$ (from `file_2_view_0` columns, e.g., "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT").

##### Parameters

- $d_j$: Demand for store $j$ (from `file_0_view_0`, column "demand", indexed by "Customer").
- $f_i$: Fixed cost to activate supplier $i$ (from `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 1").
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row "Unnamed: 0" for $i$, column for $j$).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand Satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier Activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i
   \]
   where $M$ is a sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$.

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- $I$ (suppliers): All unique values in `file_1_view_0` column "Unnamed: 1" and `file_2_view_0` row "Unnamed: 0".
- $J$ (stores): All unique values in `file_2_view_0` columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"].
- $d_j$: `file_0_view_0`, column "demand", indexed by "Customer".
- $f_i$: `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 1".
- $c_{ij}$: `file_2_view_0`, row "Unnamed: 0" for $i$, columns as above for $j$.
- $M$: $M = \sum_{j \in J} d_j$, with $d_j$ from `file_0_view_0`.

All indices, parameters, and mappings are defined directly from the CSV data as described above.