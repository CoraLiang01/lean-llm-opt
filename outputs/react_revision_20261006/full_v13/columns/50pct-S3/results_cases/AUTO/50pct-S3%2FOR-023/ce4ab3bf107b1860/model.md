##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

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
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since no explicit supplier capacity is given).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers (from `fixed_cost.csv` and rows of `transportation_costs.csv`)
- $J$: Set of stores (from `demand.csv` and columns of `transportation_costs.csv`)
- $d_j$: Demand for store $j$ (from `demand.csv`, column `demand`, indexed by `Customer`)
- $f_i$: Fixed cost for supplier $i$ (from `fixed_cost.csv`, column `fixed_costs`, indexed by `Unnamed: 1`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `transportation_costs.csv`, row `Unnamed: 0` for supplier, column for store)
- $M_i$: Big-M upper bound for supplier $i$ (set as $\sum_{j \in J} d_j$)

##### Data Mapping

- $I$: All unique values in `fixed_cost.csv`, column `Unnamed: 1` and `transportation_costs.csv`, column `Unnamed: 0` (supplier names)
- $J$: All unique values in `demand.csv`, column `Customer` and `transportation_costs.csv` columns (store names)
- $d_j$: `demand.csv`, column `demand`, indexed by `Customer` (table_id: file_0_view_0)
- $f_i$: `fixed_cost.csv`, column `fixed_costs`, indexed by `Unnamed: 1` (table_id: file_1_view_0)
- $c_{ij}$: `transportation_costs.csv`, value at row with `Unnamed: 0` = $i$, column = $j$ (table_id: file_2_view_0)
- $M_i$: $\sum_{j \in J} d_j$ (sum over all $d_j$ from `demand.csv`)

All index sets and parameters are defined exactly as present in the source data. No values are enumerated here; see the Observation for the full data.