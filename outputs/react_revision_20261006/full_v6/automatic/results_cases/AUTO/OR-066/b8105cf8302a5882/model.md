##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

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
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i = \sum_{j \in J} d_j$ is a sufficiently large constant (total demand), ensuring that if $y_i = 0$, then $x_{ij} = 0$ for all $j$.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Data Mapping

- $I$: set of suppliers, from column `Unnamed: 0` in `file_1_view_0` (`fixed_cost.csv`)
- $J$: set of supermarkets, from column `customer` in `file_0_view_0` (`demand.csv`)
- $d_j$: demand of supermarket $j$, from column `demand` in `file_0_view_0` (`demand.csv`)
- $f_i$: fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: per-unit transportation cost from supplier $i$ to supermarket $j$, from row `Unnamed: 0` = $i$, column $j$ in `file_2_view_0` (`transportation_costs.csv`)
- $M_i = \sum_{j \in J} d_j$ (total demand, computed from `demand.csv`)

##### Data Mapping

- Suppliers $I$: `file_1_view_0`, column `Unnamed: 0`
- Supermarkets $J$: `file_0_view_0`, column `customer`
- Demands $d_j$: `file_0_view_0`, columns `customer`, `demand`
- Fixed costs $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- Transportation costs $c_{ij}$: `file_2_view_0`, rows `Unnamed: 0` (supplier), columns `C1`, `C2` (supermarket)
- $M_i$: computed as $\sum_{j \in J} d_j$ from `file_0_view_0`, column `demand`