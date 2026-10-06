##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Sets

- $I$: Set of suppliers (from `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`)
- $J$: Set of stores (from `file_2_view_0` columns: `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`)

##### Parameters

- $d_j$: Demand at store $j$ (from `file_0_view_0.demand`, mapped to stores in $J$)
- $f_i$: Fixed cost for opening supplier $i$ (from `file_1_view_0.fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row: supplier $i$, column: store $j$)
- $M$: A sufficiently large constant, e.g., $M = \sum_{j \in J} d_j$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All supplier names from `file_1_view_0.Unnamed: 0` and `file_2_view_0.Unnamed: 0`
- $J$: All store names from `file_2_view_0` columns (excluding `Unnamed: 0`)
- $d_j$: `file_0_view_0.demand`, mapped to stores in $J$ (mapping between `Customer` and store names as per business logic)
- $f_i$: `file_1_view_0.fixed_costs`, indexed by supplier $i$
- $c_{ij}$: `file_2_view_0`, row: supplier $i$, column: store $j$
- $M$: $\sum_{j \in J} d_j$ (sum of all demands from `file_0_view_0.demand`)

**Note:** All index sets and parameters are defined directly from the CSV data as described above. No values are enumerated here; all mappings are symbolic and refer to the exact table and column names.