##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
$$

##### Constraints

1. Store demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Supplier activation logic:
   $$
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   $$
   where $M = \sum_{j \in J} d_j$ (a valid upper bound; no explicit supplier capacity).
3. Variable domains:
   $$
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   $$

##### Index Sets and Parameters

- $I$: Set of suppliers, from column "Unnamed: 0" in `fixed_cost.csv` and row "Unnamed: 0" in `transportation_costs.csv`.
- $J$: Set of stores, from column "Customer" in `demand.csv` and columns in `transportation_costs.csv` (excluding "Unnamed: 0").
- $d_j$: Demand for store $j$, from column "demand" in `demand.csv`.
- $f_i$: Fixed cost for supplier $i$, from column "fixed_costs" in `fixed_cost.csv`.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to store $j$, from `transportation_costs.csv` (row "Unnamed: 0" = supplier, column = store).
- $M = \sum_{j \in J} d_j$.

##### Data Mapping

- $I$: All values in `fixed_cost.csv`["Unnamed: 0"] and `transportation_costs.csv`["Unnamed: 0"].
- $J$: All values in `demand.csv`["Customer"] and `transportation_costs.csv`[columns excluding "Unnamed: 0"].
- $d_j$: `demand.csv` with mapping: $j$ = "Customer", $d_j$ = "demand". Table ID: file_0_view_0.
- $f_i$: `fixed_cost.csv` with mapping: $i$ = "Unnamed: 0", $f_i$ = "fixed_costs". Table ID: file_1_view_0.
- $c_{ij}$: `transportation_costs.csv` with mapping: $i$ = "Unnamed: 0", $j$ = each column except "Unnamed: 0". Table ID: file_2_view_0.
- $M$: $M = \sum_{j \in J} d_j$ using all $d_j$ from `demand.csv`.

All index sets and parameters are defined directly from the CSV data as described above.