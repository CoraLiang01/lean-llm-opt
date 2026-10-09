##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Supplier activation: $x_{ij} \leq D_j y_i,\quad \forall i \in I,\, j \in J$, where $D_j$ is the demand of store $j$
3. Domains: $x_{ij} \geq 0$ (continuous); $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 2" in file_1_view_0 and file_2_view_0 (fixed_cost.csv and transportation_costs.csv)
- $J$: set of stores, from column "Customer" in file_0_view_0 (demand.csv) and columns in file_2_view_0 (transportation_costs.csv)
- $d_j$: demand of store $j$, from column "demand" in file_0_view_0 (demand.csv)
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in file_1_view_0 (fixed_cost.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$, from file_2_view_0 (transportation_costs.csv), with supplier $i$ as row "Unnamed: 2" and store $j$ as column header

##### Data Mapping

- $I$: All unique values in "Unnamed: 2" from file_1_view_0 (fixed_cost.csv) and file_2_view_0 (transportation_costs.csv)
- $J$: All unique values in "Customer" from file_0_view_0 (demand.csv) and all matching column headers in file_2_view_0 (transportation_costs.csv)
- $d_j$: file_0_view_0, column "demand", indexed by "Customer"
- $f_i$: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 2"
- $c_{ij}$: file_2_view_0, row "Unnamed: 2" (supplier), column $j$ (store)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$
- $y_i$: decision variable for each $i \in I$

All index sets, parameters, and constraints are defined directly from the current CSV data.