##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Activation logic: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (warehouses, from file_1_view_0)
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$ (musicians/bands, from file_0_view_0)
- $d_j$ is demand for $j$ from column "demand" in file_0_view_0
- $f_i$ is fixed cost for $i$ from column "fixed_costs" in file_1_view_0
- $c_{ij}$ is transportation cost from $i$ to $j$ from file_2_view_0, row "Unnamed: 0" (warehouse), column $j$
- $M = \sum_{j \in J} d_j$ (total demand, valid upper bound for each warehouse)

##### Data Mapping

- $I$: All values in "Unnamed: 0" of file_1_view_0 (fixed_cost.csv)
- $J$: All values in "customer" of file_0_view_0 (demand.csv)
- $d_j$: "demand" column in file_0_view_0, indexed by "customer"
- $f_i$: "fixed_costs" column in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0 (transportation_costs.csv), row "Unnamed: 0" (warehouse), column $j$ (customer)
- $M$: $\sum_{j \in J} d_j$ from file_0_view_0

All indices and parameters are defined by the full set of current records in the respective files.