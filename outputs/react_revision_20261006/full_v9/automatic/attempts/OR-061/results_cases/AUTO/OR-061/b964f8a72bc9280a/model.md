##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: Set of suppliers, from `file_1_view_0`, column `Unnamed: 0`
- $J$: Set of branches, from `file_0_view_0`, column `customer`
- $d_j$: Demand of branch $j$, from `file_0_view_0`, column `demand`
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0`, column `fixed_costs`
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from `file_2_view_0`, row `Unnamed: 0` (supplier), column $j$ (branch/customer)
- $M_i$: For each $i$, $M_i = \sum_{j \in J} d_j$ (sum of all demands, computed from `file_0_view_0`, column `demand`)

##### Data Mapping

- Suppliers $I$: `file_1_view_0`, column `Unnamed: 0`
- Branches $J$: `file_0_view_0`, column `customer`
- Demand $d_j$: `file_0_view_0`, columns `customer`, `demand`
- Fixed cost $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- Transportation cost $c_{ij}$: `file_2_view_0`, row `Unnamed: 0`, columns `C1`, `C2`, `C3`, `C4`, `C5`
- $M_i$: $\sum_{j \in J} d_j$ from `file_0_view_0`, column `demand` (same for all $i$)

No supplier capacity limits are specified beyond activation. All indices and parameters are mapped directly from the CSV sources as described.