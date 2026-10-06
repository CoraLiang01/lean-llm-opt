##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from column `Unnamed: 0` in `file_1_view_0` (`fixed_cost.csv`).
- $J$: Set of supermarkets, from column `customer` in `file_0_view_0` (`demand.csv`).
- $d_j$: Demand of supermarket $j$, from column `demand` in `file_0_view_0` (`demand.csv`).
- $f_i$: Fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0` (`fixed_cost.csv`).
- $c_{ij}$: Per-unit transportation cost from supplier $i$ to supermarket $j$, from entry at row $i$ (`Unnamed: 1`) and column $j$ in `file_2_view_0` (`transportation_costs.csv`).
- $M$: A sufficiently large constant, $M = \sum_{j \in J} d_j$.

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

3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: All values in `file_1_view_0`, column `Unnamed: 0` (`fixed_cost.csv`)
- $J$: All values in `file_0_view_0`, column `customer` (`demand.csv`)
- $d_j$: `file_0_view_0`, columns `customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 1` (supplier), columns `C1`, `C2` (supermarkets) (`transportation_costs.csv`)
- $M$: $\sum_{j \in J} d_j$, using `file_0_view_0`, column `demand`