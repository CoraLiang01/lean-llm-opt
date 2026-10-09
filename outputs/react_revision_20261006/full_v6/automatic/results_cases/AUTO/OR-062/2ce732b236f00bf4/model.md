##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store Demand Satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each store's demand must be fully met.)

2. **Supplier Activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (A supplier can only ship if activated. $M_i$ is a sufficiently large upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 0`)
- $J$: Set of stores (from `file_2_view_0`, columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`)
- $d_j$: Demand of store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 0` = $i$, column $j$)
- $M_i$: Big-M parameter for each supplier $i$ (set as $\sum_{j \in J} d_j$)

##### Data Mapping

- Suppliers $I$: All unique values in `file_1_view_0`, column `Unnamed: 0`
- Stores $J$: All unique values in `file_2_view_0`, columns (excluding `Unnamed: 0`)
- $d_j$: For each $j \in J$, from `file_0_view_0`, column `demand`, with mapping between store names in $J$ and `Customer` in `file_0_view_0`
- $f_i$: For each $i \in I$, from `file_1_view_0`, column `fixed_costs`
- $c_{ij}$: For each $i \in I$, $j \in J$, from `file_2_view_0`, row where `Unnamed: 0` = $i$, column $j$
- $M_i$: For each $i \in I$, $M_i = \sum_{j \in J} d_j$ (sum over all $d_j$ from `file_0_view_0`)

All index sets and parameters are defined exactly as present in the source data. No values are enumerated here; see the Observation for all current data.