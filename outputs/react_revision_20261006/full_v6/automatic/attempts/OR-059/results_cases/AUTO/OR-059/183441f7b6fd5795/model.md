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
2. **Supplier activation logic:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ is a sufficiently large constant (total demand).
3. **Variable domains:**  
   \[
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` (`file_1_view_0`)
- $J$: Set of dealerships, from column "customer" in `demand.csv` (`file_0_view_0`)
- $d_j$: Demand of dealership $j$, from column "demand" in `demand.csv` (`file_0_view_0`)
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv` (`file_1_view_0`)
- $c_{ij}$: Transportation cost per vehicle from supplier $i$ to dealership $j$, from matrix in `transportation_costs.csv` (`file_2_view_0`), with rows indexed by "Unnamed: 0" (suppliers) and columns by dealership IDs ("C1", ..., "C9")
- $M$: $\sum_{j \in J} d_j$ (total demand, computed from all $d_j$)

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of `file_1_view_0` (fixed_cost.csv)
- $J$: All values in column "customer" of `file_0_view_0` (demand.csv)
- $d_j$: Column "demand" in `file_0_view_0` (demand.csv), keyed by "customer"
- $f_i$: Column "fixed_costs" in `file_1_view_0` (fixed_cost.csv), keyed by "Unnamed: 0"
- $c_{ij}$: Matrix in `file_2_view_0` (transportation_costs.csv), rows "Unnamed: 0" (suppliers), columns "C1"..."C9" (dealerships)
- $M$: $\sum_{j \in J} d_j$ using all "demand" values in `file_0_view_0`

No additional capacity or proportion constraints are imposed beyond those above. All indices and parameters are defined directly from the current CSV data.