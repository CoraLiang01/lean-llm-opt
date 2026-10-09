##### Decision Variables

- $x_{ij} \geq 0$: Quantity of liquor product shipped from supplier (facility) $i \in I$ to store (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to activate supplier $i$.

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand at store $j$.

2. **Supplier activation logic:**
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   where $M_i$ is a sufficiently large upper bound for supplier $i$ (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Index Sets and Data Mapping

- $I$: Set of suppliers (facilities), from `file_1_view_0`, column `Unnamed: 3`.
- $J$: Set of stores (customers), from `file_0_view_0`, column `Customer`.
- $d_j$: Demand at store $j$, from `file_0_view_0`, column `demand`.
- $f_i$: Fixed cost for supplier $i$, from `file_1_view_0`, column `fixed_costs`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `file_2_view_0`, row identifier `Unnamed: 4` (supplier), column header (store).
- $M_i$: For each $i$, $M_i = \sum_{j \in J} d_j$ (sum of all demands).

##### Data Mapping

- $I$: `file_1_view_0`, column `Unnamed: 3`
- $J$: `file_0_view_0`, column `Customer`
- $d_j$: `file_0_view_0`, columns `Customer`, `demand`
- $f_i$: `file_1_view_0`, columns `Unnamed: 3`, `fixed_costs`
- $c_{ij}$: `file_2_view_0`, row `Unnamed: 4` (supplier), columns with store names (see `file_2_view_0` for mapping)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (computed from all $d_j$ above)

No additional capacity or resource constraints are imposed unless specified in the data. All variables and parameters are mapped directly from the provided CSV files.