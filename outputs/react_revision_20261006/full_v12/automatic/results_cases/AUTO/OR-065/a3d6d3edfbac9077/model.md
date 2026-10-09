##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I} x_{ij} = d_j,\quad \forall j \in J$
2. Activation logic: $\sum_{j\in J} x_{ij} \leq M y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}\}$ (warehouses, from file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0)
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$ (musicians/bands, from file_0_view_0.customer and file_2_view_0 columns)
- $d_j$ is demand for $j$ from file_0_view_0.demand
- $f_i$ is fixed cost for $i$ from file_1_view_0.fixed_costs
- $c_{ij}$ is transportation cost from $i$ to $j$ from file_2_view_0, with rows indexed by Unnamed: 0 and columns by C1, C2, C3
- $M = \sum_{j \in J} d_j$ (total demand, a valid upper bound for activation logic)

##### Data Mapping

- $I$: file_1_view_0.Unnamed: 0 and file_2_view_0.Unnamed: 0
- $J$: file_0_view_0.customer and file_2_view_0 columns [C1, C2, C3]
- $d_j$: file_0_view_0.demand
- $f_i$: file_1_view_0.fixed_costs
- $c_{ij}$: file_2_view_0, rows Unnamed: 0, columns C1, C2, C3
- $M$: $\sum_{j \in J} d_j$ from file_0_view_0.demand

All indices, parameters, and mappings are preserved exactly as in the source data.