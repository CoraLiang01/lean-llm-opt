##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to customer (store) $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (where $D_j$ is the demand of customer $j$)
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets

$I$ = set of suppliers (facility locations) from column "Unnamed: 3" in file_1_view_0 (fixed_cost.csv) and "Unnamed: 4" in file_2_view_0 (transportation_costs.csv)

$J$ = set of customers (stores) from column "Customer" in file_0_view_0 (demand.csv) and columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] in file_2_view_0 (transportation_costs.csv)

##### Parameters and Data Mapping

- $d_j$: demand for customer $j$ from column "demand" in file_0_view_0, indexed by "Customer"
- $f_i$: fixed cost for supplier $i$ from column "fixed_costs" in file_1_view_0, indexed by "Unnamed: 3"
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ from file_2_view_0, with rows indexed by "Unnamed: 4" (supplier) and columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] (customer)
- $D_j$: demand for customer $j$ (same as $d_j$), used for activation constraint

##### Data Mapping

- Suppliers $I$: file_1_view_0["Unnamed: 3"], file_2_view_0["Unnamed: 4"]
- Customers $J$: file_0_view_0["Customer"], file_2_view_0 columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"]
- Demand $d_j$: file_0_view_0["demand"], indexed by file_0_view_0["Customer"]
- Fixed cost $f_i$: file_1_view_0["fixed_costs"], indexed by file_1_view_0["Unnamed: 3"]
- Transportation cost $c_{ij}$: file_2_view_0, rows "Unnamed: 4" (supplier), columns ["BANCROFT", "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO"] (customer)

All index sets and parameters are defined by the full set of entities present in the respective columns of the current CSV files. No values are enumerated here; see the Observation for all current data.