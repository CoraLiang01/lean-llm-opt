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

- $I$: Set of suppliers (from `fixed_cost.csv` and rows of `transportation_costs.csv`),  
  $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J$: Set of stores (from `demand.csv` and columns of `transportation_costs.csv`),  
  $J = \{\text{CLARINDA}, \text{FORT MADISON}, \text{SIOUX CITY}, \text{TOLEDO}, \text{BANCROFT}\}$
- $d_j$: Demand for store $j$ (from `demand.csv`, column `demand`, indexed by `Customer`)
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`, column `fixed_costs`, indexed by `Unnamed: 1`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`, row `Unnamed: 0` for $i$, column $j$)
- $M_i$: Big-M for each supplier $i$ (set as $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- $I$ (suppliers):  
  - Table: `file_1_view_0` (`fixed_cost.csv`), column: `Unnamed: 1`
  - Table: `file_2_view_0` (`transportation_costs.csv`), row: `Unnamed: 0`
- $J$ (stores):  
  - Table: `file_0_view_0` (`demand.csv`), column: `Customer`
  - Table: `file_2_view_0` (`transportation_costs.csv`), columns: `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`
- $d_j$:  
  - Table: `file_0_view_0` (`demand.csv`), column: `demand`, indexed by `Customer`
- $f_i$:  
  - Table: `file_1_view_0` (`fixed_cost.csv`), column: `fixed_costs`, indexed by `Unnamed: 1`
- $c_{ij}$:  
  - Table: `file_2_view_0` (`transportation_costs.csv`), row: `Unnamed: 0` for $i$, columns: as above for $j$
- $M_i$:  
  - $M_i = \sum_{j \in J} d_j$ (sum over all $d_j$ from `file_0_view_0`)

All index sets and parameters are defined directly from the current CSV data. No values are enumerated here; see the Observation for all raw data.