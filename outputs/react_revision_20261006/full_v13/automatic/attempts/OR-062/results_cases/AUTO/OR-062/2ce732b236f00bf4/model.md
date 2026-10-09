##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

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
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a valid upper bound on total shipments from any supplier (since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and row "Unnamed: 0" in `transportation_costs.csv`.
- $J$: Set of stores, from column "Customer" in `demand.csv` and columns in `transportation_costs.csv` (excluding "Unnamed: 0").
- $d_j$: Demand of store $j$, from column "demand" in `demand.csv`.
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `transportation_costs.csv` (row "Unnamed: 0" = $i$, column $j$).
- $M = \sum_{j \in J} d_j$, computed from all $d_j$.

##### Data Mapping

- $I$: All values in `fixed_cost.csv` column "Unnamed: 0" and `transportation_costs.csv` row "Unnamed: 0" (must match).
- $J$: All values in `demand.csv` column "Customer" and `transportation_costs.csv` columns (excluding "Unnamed: 0").
- $d_j$: `demand.csv`, table_id: file_0_view_0, columns: "Customer", "demand".
- $f_i$: `fixed_cost.csv`, table_id: file_1_view_0, columns: "Unnamed: 0", "fixed_costs".
- $c_{ij}$: `transportation_costs.csv`, table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$.
- $M$: $\sum_{j \in J} d_j$, using all $d_j$ from `demand.csv`.

All index sets and parameters are defined by the full set of current records in the respective CSV files.