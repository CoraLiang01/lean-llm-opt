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
2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large constant (the total demand), since no explicit supplier capacity is given.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `fixed_cost.csv` and `transportation_costs.csv` rows; see Data Mapping).
- $J$: Set of stores (from `demand.csv` and `transportation_costs.csv` columns; see Data Mapping).
- $d_j$: Demand for store $j$ (from `demand.csv`).
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`).

##### Data Mapping

- $I$ (Suppliers):  
  - Table: `file_1_view_0` (`fixed_cost.csv`), column: `Unnamed: 1`  
  - Table: `file_2_view_0` (`transportation_costs.csv`), row: `Unnamed: 0`
- $J$ (Stores):  
  - Table: `file_0_view_0` (`demand.csv`), column: `Customer`  
  - Table: `file_2_view_0` (`transportation_costs.csv`), columns: `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`
- $d_j$:  
  - Table: `file_0_view_0` (`demand.csv`), columns: `Customer`, `demand`
- $f_i$:  
  - Table: `file_1_view_0` (`fixed_cost.csv`), columns: `Unnamed: 1`, `fixed_costs`
- $c_{ij}$:  
  - Table: `file_2_view_0` (`transportation_costs.csv`), rows: `Unnamed: 0` (supplier), columns: `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` (store)
- $M_i$:  
  - $M_i = \sum_{j \in J} d_j$, where $d_j$ is from `file_0_view_0` (`demand.csv`)

All index sets and parameters are defined by the full set of entities present in the respective CSV files as described above.