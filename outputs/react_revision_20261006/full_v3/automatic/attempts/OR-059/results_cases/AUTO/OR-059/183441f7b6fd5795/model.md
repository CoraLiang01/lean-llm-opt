##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: transportation cost per vehicle from supplier $i$ to dealership $j$,
- $f_i$: fixed cost to open supplier $i$.

##### Constraints

1. **Dealership Demand Satisfaction**:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand of dealership $j$.

2. **Supplier Activation**:
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large constant (e.g., $M_i = \sum_{j \in J} d_j$) to ensure that if $y_i = 0$, then $x_{ij} = 0$ for all $j$.

3. **Variable Domains**:
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets

- $I$: Set of suppliers, from `fixed_cost.csv` column `Unnamed: 0` in table_id `file_1_view_0`.
- $J$: Set of dealerships, from `demand.csv` column `customer` in table_id `file_0_view_0`.

##### Data Mapping

- $d_j$: Demand for dealership $j$, from `demand.csv` column `demand` in table_id `file_0_view_0`, indexed by `customer`.
- $f_i$: Fixed cost for supplier $i$, from `fixed_cost.csv` column `fixed_costs` in table_id `file_1_view_0`, indexed by `Unnamed: 0`.
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from `transportation_costs.csv` in table_id `file_2_view_0`, with rows indexed by `Unnamed: 0` (supplier) and columns by dealership IDs (`C1`, ..., `C9`).
- $M_i$: For each $i$, $M_i = \sum_{j \in J} d_j$ (sum of all dealership demands).

##### Matrix Structure

- The transportation cost matrix $[c_{ij}]$ has suppliers as rows (`Unnamed: 0` in `file_2_view_0`) and dealerships as columns (`C1` to `C9` in `file_2_view_0`), aligned with the index sets above.