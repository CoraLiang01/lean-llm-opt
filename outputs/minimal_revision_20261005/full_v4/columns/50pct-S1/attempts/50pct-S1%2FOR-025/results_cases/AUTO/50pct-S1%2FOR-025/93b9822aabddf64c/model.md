##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
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
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers (from `file_1_view_0`, column `Unnamed: 0`)
- $J$: Set of supermarkets (from `file_0_view_0`, column `customer`)

##### Parameters and Data Mapping

- $d_j$: Demand of supermarket $j$  
  — Source: `file_0_view_0`, columns: `customer`, `demand`
- $f_i$: Fixed cost for activating supplier $i$  
  — Source: `file_1_view_0`, columns: `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$  
  — Source: `file_2_view_0`, rows: `Unnamed: 0` (supplier), columns: `C1`, `C2` (supermarkets)
- $M$: Big-M parameter, $M = \sum_{j \in J} d_j$ (sum of all supermarket demands from `file_0_view_0`, column `demand`)

##### Data Mapping

- **Suppliers ($I$):**  
  `file_1_view_0`, column `Unnamed: 0`
- **Supermarkets ($J$):**  
  `file_0_view_0`, column `customer`
- **Demands ($d_j$):**  
  `file_0_view_0`, columns `customer`, `demand`
- **Fixed costs ($f_i$):**  
  `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- **Transportation costs ($c_{ij}$):**  
  `file_2_view_0`, rows `Unnamed: 0`, columns `C1`, `C2`

All index sets and parameters are defined directly from the CSV data as described above.