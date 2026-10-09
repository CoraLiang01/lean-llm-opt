##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Supermarket demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Supplier activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (a valid upper bound on total shipments from any supplier, since there are no explicit supplier capacity limits).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of suppliers, from column `Unnamed: 0` in `file_1_view_0` (`fixed_cost.csv`)
- $J$: set of supermarkets, from column `customer` in `file_0_view_0` (`demand.csv`)
- $d_j$: demand of supermarket $j$, from column `demand` in `file_0_view_0` (`demand.csv`)
- $f_i$: fixed cost for supplier $i$, from column `fixed_costs` in `file_1_view_0` (`fixed_cost.csv`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to supermarket $j$, from matrix in `file_2_view_0` (`transportation_costs.csv`), with rows indexed by `Unnamed: 0` (supplier) and columns by supermarket IDs.
- $M = \sum_{j \in J} d_j$ (sum of all supermarket demands, using column `demand` in `file_0_view_0`)

##### Data Mapping

- $I$: All values in `file_1_view_0`, column `Unnamed: 0`
- $J$: All values in `file_0_view_0`, column `customer`
- $d_j$: `file_0_view_0`, columns `customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 0`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, rows indexed by `Unnamed: 0` (supplier), columns by supermarket IDs (`C1`, ..., `C25`)
- $M$: $\sum_{j \in J} d_j$, using all `demand` values in `file_0_view_0`