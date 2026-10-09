##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Store demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large constant for each supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (facility locations), from `file_1_view_0` column `Unnamed: 1` and `file_2_view_0` column `Unnamed: 0`.
- $J$: Set of stores (customers), from `file_0_view_0` column `Customer` and `file_2_view_0` columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- $d_j$: Demand for store $j$, from `file_0_view_0` columns `Customer`, `demand`.
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0` columns `Unnamed: 1`, `fixed_costs`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` rows `Unnamed: 0` (supplier), columns (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`).
- $M_i$: Big-M for each supplier $i$, set as $M_i = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All unique values in `file_1_view_0` column `Unnamed: 1` and `file_2_view_0` column `Unnamed: 0`.
- $J$: All unique values in `file_0_view_0` column `Customer` and `file_2_view_0` columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- $d_j$: `file_0_view_0` columns `Customer`, `demand`.
- $f_i$: `file_1_view_0` columns `Unnamed: 1`, `fixed_costs`.
- $c_{ij}$: `file_2_view_0` rows `Unnamed: 0` (supplier), columns (`CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`).
- $M_i$: $M_i = \sum_{j \in J} d_j$, with $d_j$ from `file_0_view_0`.

All index sets and parameters are defined by the full set of current records in the respective columns of the CSV files. No values are omitted or abbreviated.