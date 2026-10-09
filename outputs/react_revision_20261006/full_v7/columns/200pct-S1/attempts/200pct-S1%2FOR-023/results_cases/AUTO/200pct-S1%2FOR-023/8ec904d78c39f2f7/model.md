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

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 3`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)
- $d_j$: Demand of store $j$ (from `file_0_view_0`, column `demand`)
- $f_i$: Fixed cost for supplier $i$ (from `file_1_view_0`, column `fixed_costs`)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$ (from `file_2_view_0`, row `Unnamed: 4` for supplier, column header for store)
- $M$: $\sum_{j \in J} d_j$ (sum of all demands)

##### Data Mapping

- $I$: All values in `file_1_view_0`, column `Unnamed: 3`
- $J$: All values in `file_0_view_0`, column `Customer`
- $d_j$: `file_0_view_0`, columns `Customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 3`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, rows indexed by `Unnamed: 4` (supplier), columns by store name (see below)
- $M$: $\sum_{j \in J} d_j$ (from all `demand` in `file_0_view_0`)

**Note:**  
- The mapping between store names in `file_0_view_0` (`Customer_1`, ...) and the columns in `file_2_view_0` (`BANCROFT`, `CLARINDA`, etc.) must be established according to the actual data dictionary or metadata. For this model, $c_{ij}$ is defined as the entry in `file_2_view_0` at row with `Unnamed: 4 = i$ (supplier), column with header $j$ (store).
- All index sets and parameters are to be taken from the full, current CSV data as described above.