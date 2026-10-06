##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers, from `file_1_view_0`, column `Unnamed: 0`
- $J$: Set of supermarkets, from `file_0_view_0`, column `customer`

##### Parameters and Data Mapping

- $d_j$: Demand of supermarket $j$, from `file_0_view_0`, column `demand`, indexed by `customer`
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0`, column `fixed_costs`, indexed by `Unnamed: 0`
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from `file_2_view_0`, row `Unnamed: 0` (supplier), column $j$ (supermarket, matching `customer`)
- $M = \sum_{j \in J} d_j$, with $d_j$ as above

##### Data Mapping

- $I$: `file_1_view_0`, column `Unnamed: 0`
- $J$: `file_0_view_0`, column `customer`
- $d_j$: `file_0_view_0`, columns `customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 0` (supplier), columns matching $j$ in `customer`
- $M$: $\sum_{j \in J} d_j$ from `file_0_view_0`, column `demand`