##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$ (inactive suppliers cannot ship)
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I$ = set of suppliers (from file_1_view_0: "Unnamed: 0")
- $J$ = set of stores (from file_2_view_0: columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT")
- $d_j$ = demand for store $j$ (from file_0_view_0: "Customer", "demand")
- $f_i$ = fixed cost for supplier $i$ (from file_1_view_0: "Unnamed: 0", "fixed_costs")
- $c_{ij}$ = transportation cost per unit from supplier $i$ to store $j$ (from file_2_view_0: row "Unnamed: 0", column $j$)
- $D_j$ = demand for store $j$ (from file_0_view_0: "Customer", "demand")

##### Data Mapping

- Suppliers $I$: file_1_view_0, column "Unnamed: 0"
- Stores $J$: file_2_view_0, columns "CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"
- Demand $d_j$: file_0_view_0, columns "Customer", "demand"
- Fixed cost $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 0" (supplier), column (store)
- $x_{ij}$, $y_i$: as defined above

Note: If the mapping between "Customer" in demand.csv and store columns in transportation_costs.csv is not one-to-one, align $J$ as the set of store columns in transportation_costs.csv, and $d_j$ as the demand for each corresponding store. If further mapping is needed, clarify the correspondence.