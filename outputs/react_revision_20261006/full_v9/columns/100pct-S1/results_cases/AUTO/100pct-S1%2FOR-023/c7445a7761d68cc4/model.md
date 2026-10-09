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
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (the total demand).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `fixed_cost.csv` and rows of `transportation_costs.csv`), indexed by the values in column `Unnamed: 2` of `file_1_view_0` and `file_2_view_0`.
- $J$: Set of stores (from `demand.csv` and columns of `transportation_costs.csv`), indexed by the values in column `Customer` of `file_0_view_0` and columns `BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO` of `file_2_view_0`.
- $d_j$: Demand for store $j$, from column `demand` in `file_0_view_0`.
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from the intersection of row `Unnamed: 2` (supplier) and columns `BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO` in `file_2_view_0$.
- $M = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All unique values in `file_1_view_0`.`Unnamed: 2` and `file_2_view_0`.`Unnamed: 2` (supplier names).
- $J$: All unique values in `file_0_view_0`.`Customer` and columns `BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO` in `file_2_view_0`.
- $d_j$: `file_0_view_0` (column: `demand`, key: `Customer`).
- $f_i$: `file_1_view_0` (column: `fixed_costs`, key: `Unnamed: 2`).
- $c_{ij}$: `file_2_view_0` (row: `Unnamed: 2` for supplier $i$, column: store $j$).
- $M$: $\sum_{j \in J} d_j$ (sum over all `demand` in `file_0_view_0`).

All index sets and parameters are defined exactly as present in the source data. No values are enumerated here; see the Observation for all current values.