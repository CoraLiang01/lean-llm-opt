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
   where $M = \sum_{j \in J} d_j$ (a valid upper bound; no explicit supplier capacity).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 3`)
- $J$: Set of stores (from `file_0_view_0`, column `Customer`)

##### Parameters and Data Mapping

- $d_j$: Demand for store $j$  
  — Source: `file_0_view_0`, columns:  
    - Store index: `Customer`  
    - Demand: `demand`
- $f_i$: Fixed cost for supplier $i$  
  — Source: `file_1_view_0`, columns:  
    - Supplier index: `Unnamed: 3`  
    - Fixed cost: `fixed_costs`
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$  
  — Source: `file_2_view_0`, columns:  
    - Supplier index: `Unnamed: 4`  
    - Store index: column names: `BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`
- $M$: $M = \sum_{j \in J} d_j$ (sum of all store demands; computed from `file_0_view_0`, column `demand`)

##### Data Mapping

- Suppliers ($I$):  
  - Table: `file_1_view_0`, column: `Unnamed: 3`
- Stores ($J$):  
  - Table: `file_0_view_0`, column: `Customer`
- Demand ($d_j$):  
  - Table: `file_0_view_0`, columns: `Customer`, `demand`
- Fixed cost ($f_i$):  
  - Table: `file_1_view_0`, columns: `Unnamed: 3`, `fixed_costs`
- Transportation cost ($c_{ij}$):  
  - Table: `file_2_view_0`, rows: `Unnamed: 4` (supplier), columns: store names (`BANCROFT`, `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`)
- $M$:  
  - $M = \sum_{j \in J} d_j$, with $d_j$ from `file_0_view_0`, column `demand`

##### Notes

- All suppliers and stores present in the respective CSVs are included in $I$ and $J$.
- The transportation cost matrix $c_{ij}$ is mapped by matching supplier names (`Unnamed: 4`) to store columns in `file_2_view_0`.
- There are no explicit supplier capacity constraints; $M$ is a valid upper bound for activation logic.