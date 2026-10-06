##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all dealerships).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers, from `file_1_view_0`, column `Unnamed: 0`
- $J$: Set of dealerships, from `file_0_view_0`, column `customer`

##### Parameters and Data Mapping

- $d_j$: Demand of dealership $j$  
  — from `file_0_view_0`, columns: `customer`, `demand`
- $f_i$: Fixed cost for opening supplier $i$  
  — from `file_1_view_0`, columns: `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$  
  — from `file_2_view_0`, rows: `Unnamed: 0` (supplier), columns: dealership IDs (`C1`, ..., `C9`)
- $M$: $\sum_{j \in J} d_j$ (total demand), computed from all $d_j$ in `file_0_view_0`

##### Data Mapping

- Suppliers ($I$): `file_1_view_0`, column `Unnamed: 0`
- Dealerships ($J$): `file_0_view_0`, column `customer`
- Demand ($d_j$): `file_0_view_0`, columns `customer`, `demand`
- Fixed cost ($f_i$): `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- Transportation cost ($c_{ij}$): `file_2_view_0`, row `Unnamed: 0` (supplier), columns `C1`–`C9` (dealerships)
- $M$: $\sum_{j \in J} d_j$ from all rows in `file_0_view_0`, column `demand`