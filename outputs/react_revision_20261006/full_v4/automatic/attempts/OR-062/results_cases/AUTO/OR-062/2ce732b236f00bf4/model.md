##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to store $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise, for all $i \in I$.

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
   where $M = \sum_{j \in J} d_j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from `file_1_view_0` column `Unnamed: 0` and `file_2_view_0` column `Unnamed: 0`.
- $J$: set of stores, from `file_0_view_0` column `Customer` and `file_2_view_0` columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- $d_j$: demand of store $j$, from `file_0_view_0` columns `Customer`, `demand`.
- $f_i$: fixed cost for supplier $i$, from `file_1_view_0` columns `Unnamed: 0`, `fixed_costs`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0` rows `Unnamed: 0` (suppliers), columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` (stores).
- $M = \sum_{j \in J} d_j$.

##### Data Mapping

- Supplier index set $I$: all unique values in `file_1_view_0` column `Unnamed: 0` and `file_2_view_0` column `Unnamed: 0`.
- Store index set $J$: all unique values in `file_0_view_0` column `Customer` and `file_2_view_0` columns `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT`.
- Demand $d_j$: from `file_0_view_0`, mapping `Customer` to `demand`.
- Fixed cost $f_i$: from `file_1_view_0`, mapping `Unnamed: 0` to `fixed_costs`.
- Transportation cost $c_{ij}$: from `file_2_view_0`, mapping row `Unnamed: 0` (supplier) and column (store) to value.
- $M$: sum of all $d_j$ from `file_0_view_0` column `demand`.