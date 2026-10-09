##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Customer demand:  
   $\sum_{i \in I} x_{ij} = d_j,\quad \forall j \in J$

2. Plant capacity:  
   $\sum_{j \in J} x_{ij} \leq K_i y_i,\quad \forall i \in I$

3. Domains:  
   $x_{ij} \geq 0$ (continuous), $y_i \in \{0,1\}$

##### Index Sets and Parameters

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from file_0_view_0, column facility_id)
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from file_1_view_0, column customer_id)
- $f_i$: fixed opening cost for plant $i$ (from file_0_view_0, column fixed_opening_cost)
- $K_i$: capacity of plant $i$ (from file_0_view_0, column facility_capacity)
- $d_j$: demand of customer $j$ (from file_1_view_0, column demand_units)
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$ (from file_0_view_0, column transportation_cost_to_Cj, where $j$ matches customer_id in file_1_view_0)

##### Data Mapping

- Plants $I$: file_0_view_0, column facility_id
- Customers $J$: file_1_view_0, column customer_id
- Fixed opening cost $f_i$: file_0_view_0, column fixed_opening_cost
- Plant capacity $K_i$: file_0_view_0, column facility_capacity
- Customer demand $d_j$: file_1_view_0, column demand_units
- Transportation cost $c_{ij}$: file_0_view_0, columns transportation_cost_to_C1 ... transportation_cost_to_C15, indexed by plant $i$ and customer $j$