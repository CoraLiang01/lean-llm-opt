##### Decision Variables

$x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
$y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise.

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Dealership demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Supplier activation logic:
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   $$
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since no supplier capacity is specified).
3. Variable domains:
   $$
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   $$

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in table_id "file_1_view_0" and "file_2_view_0".
- $J$: Set of dealerships, from column "customer" in table_id "file_0_view_0" and columns "C1"..."C9" in "file_2_view_0".
- $d_j$: Demand of dealership $j$, from column "demand" in table_id "file_0_view_0".
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in table_id "file_1_view_0".
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from table_id "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M_i$: Big-M upper bound for supplier $i$, set as $\sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All values in "Unnamed: 0" column of "file_1_view_0" and "file_2_view_0".
- $J$: All values in "customer" column of "file_0_view_0" and columns "C1"..."C9" of "file_2_view_0".
- $d_j$: "demand" column in "file_0_view_0", indexed by "customer".
- $f_i$: "fixed_costs" column in "file_1_view_0", indexed by "Unnamed: 0".
- $c_{ij}$: Table "file_2_view_0", row "Unnamed: 0" = $i$, column $j$.
- $M_i$: $\sum_{j \in J} d_j$ (sum over all "demand" in "file_0_view_0").

All parameters and sets are to be taken exactly as defined in the source tables.