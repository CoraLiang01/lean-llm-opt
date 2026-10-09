##### Decision Variables

$x_{ij} \geq 0$: quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$
2. Supplier activation (no shipment from inactive suppliers):
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   $$
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since there are no explicit supplier capacity limits).
3. Variable domains:
   $$
   x_{ij} \geq 0, \quad y_i \in \{0,1\}
   $$

##### Index Sets and Parameters

- $I$: set of suppliers, from column "Unnamed: 1" in file_1_view_0 and "Unnamed: 0" in file_2_view_0.
- $J$: set of stores, from column "Customer" in file_0_view_0 and columns ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"] in file_2_view_0.
- $d_j$: demand for store $j$, from column "demand" in file_0_view_0.
- $f_i$: fixed cost for supplier $i$, from column "fixed_costs" in file_1_view_0.
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$, from file_2_view_0, with supplier $i$ as row "Unnamed: 0" and store $j$ as column header.
- $M_i$: upper bound for supplier $i$, set as $\sum_{j \in J} d_j$.

##### Data Mapping

- $I$: file_1_view_0["Unnamed: 1"], file_2_view_0["Unnamed: 0"]
- $J$: file_0_view_0["Customer"], file_2_view_0 column headers ["CLARINDA", "FORT MADISON", "SIOUX CITY", "TOLEDO", "BANCROFT"]
- $d_j$: file_0_view_0["demand"]
- $f_i$: file_1_view_0["fixed_costs"]
- $c_{ij}$: file_2_view_0, rows indexed by "Unnamed: 0" (supplier), columns by store name
- $M_i$: $\sum_{j \in J} d_j$ (computed from file_0_view_0["demand"])