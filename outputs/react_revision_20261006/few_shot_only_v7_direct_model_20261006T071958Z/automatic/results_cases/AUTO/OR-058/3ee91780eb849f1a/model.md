##### Decision Variables

- $x_{ij} \geq 0$: Quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$.
- $f_i$: Fixed cost to open supplier $i$.

##### Constraints

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   where $d_j$ is the demand at store $j$.

2. **Suppliers can only ship if open:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand across all stores).

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and `transportation_costs.csv`.
- $J$: Set of stores, from column "customer" in `demand.csv` and columns "C1", ..., "C6" in `transportation_costs.csv`.

##### Data Mapping

- $I$ (suppliers): All unique values in `file_1_view_0` column "Unnamed: 0" and `file_2_view_0` column "Unnamed: 0".
- $J$ (stores): All unique values in `file_0_view_0` column "customer" and `file_2_view_0` columns "C1"–"C6".
- $d_j$: From `file_0_view_0`, column "demand", indexed by "customer".
- $f_i$: From `file_1_view_0`, column "fixed_costs", indexed by "Unnamed: 0".
- $c_{ij}$: From `file_2_view_0`, row "Unnamed: 0" = $i$, column $j$.
- $M = \sum_{j \in J} d_j$, with $d_j$ as above.

All parameters are mapped directly to the supplied CSV data. No additional constraints or bounds are imposed beyond those described.