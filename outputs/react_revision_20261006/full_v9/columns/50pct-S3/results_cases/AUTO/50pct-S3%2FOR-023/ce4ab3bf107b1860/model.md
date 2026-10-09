##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

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
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in J} d_j$).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers (from `fixed_cost.csv` and rows of `transportation_costs.csv`)
- $J$: set of stores (from `demand.csv` and columns of `transportation_costs.csv`)
- $d_j$: demand for store $j$ (from `demand.csv`, column `demand`, indexed by `Customer`)
- $f_i$: fixed cost for supplier $i$ (from `fixed_cost.csv`, column `fixed_costs`, indexed by `Unnamed: 1`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`, row `Unnamed: 0` for supplier $i$, column for store $j$)
- $M_i$: big-M for each supplier $i$ (set as $\sum_{j \in J} d_j$)

##### Data Mapping

- $I$: All unique values in `fixed_cost.csv`, column `Unnamed: 1` (supplier names), and all unique values in `transportation_costs.csv`, column `Unnamed: 0` (supplier names).
- $J$: All unique values in `demand.csv`, column `Customer`, and all unique values in `transportation_costs.csv`, columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- $d_j$: From `demand.csv`, table_id `file_0_view_0`, column `demand`, indexed by `Customer`.
- $f_i$: From `fixed_cost.csv`, table_id `file_1_view_0`, column `fixed_costs`, indexed by `Unnamed: 1`.
- $c_{ij}$: From `transportation_costs.csv`, table_id `file_2_view_0`, row `Unnamed: 0` (supplier $i$), column (store $j$).
- $M_i$: $M_i = \sum_{j \in J} d_j$, with $d_j$ as above.

All index sets and parameters are defined directly from the current CSV data, with no abbreviation or omission.