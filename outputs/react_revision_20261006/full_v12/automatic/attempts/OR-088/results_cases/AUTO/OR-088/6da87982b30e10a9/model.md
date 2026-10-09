##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous)  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened)

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand: $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$
2. Plant capacity: $\sum_{j \in J} x_{ij} \leq \text{cap}_i y_i,\quad \forall i \in I$
3. Domains: $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of plants, from column "plant" in file_0_view_0 (cost.csv)
- $J$: set of customers, from column "customer" in file_1_view_0 (demand.csv)
- $f_i$: fixed_cost for plant $i$, from column "fixed_cost" in file_0_view_0
- $\text{cap}_i$: capacity for plant $i$, from column "capacity" in file_0_view_0
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$, from columns "C1"–"C15" in file_0_view_0
- $d_j$: demand for customer $j$, from column "demand" in file_1_view_0

##### Data Mapping

- Plants $I$: file_0_view_0, column "plant"
- Customers $J$: file_1_view_0, column "customer"
- Fixed costs $f_i$: file_0_view_0, column "fixed_cost"
- Plant capacities $\text{cap}_i$: file_0_view_0, column "capacity"
- Transport costs $c_{ij}$: file_0_view_0, columns "C1"–"C15" (row $i$ for plant, column $j$ for customer)
- Customer demands $d_j$: file_1_view_0, column "demand" (row $j$ for customer)

All index sets and parameters are defined exactly as in the source data.