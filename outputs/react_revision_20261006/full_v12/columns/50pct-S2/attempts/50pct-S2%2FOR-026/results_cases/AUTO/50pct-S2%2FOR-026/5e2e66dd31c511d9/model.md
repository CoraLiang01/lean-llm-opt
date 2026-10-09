##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Customer demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Plant capacity (only if built):
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
3. Variable domains:
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of plants, from column "facility_id" in file_0_view_0 (cost.csv)
- $J$: set of customers, from column "customer_id" in file_1_view_0 (demand.csv)
- $f_i$: fixed opening cost for plant $i$, from column "fixed_opening_cost" in file_0_view_0
- $\text{cap}_i$: capacity of plant $i$, from column "facility_capacity" in file_0_view_0
- $d_j$: demand of customer $j$, from column "demand_units" in file_1_view_0
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from column "transportation_cost_to_{j}" in file_0_view_0, where $j$ matches customer_id in file_1_view_0

##### Data Mapping

- Plants $I$: file_0_view_0, column "facility_id"
- Customers $J$: file_1_view_0, column "customer_id"
- Fixed opening cost $f_i$: file_0_view_0, column "fixed_opening_cost"
- Plant capacity $\text{cap}_i$: file_0_view_0, column "facility_capacity"
- Customer demand $d_j$: file_1_view_0, column "demand_units"
- Transportation cost $c_{ij}$: file_0_view_0, column "transportation_cost_to_{j}" (where $j$ is the customer_id from file_1_view_0)

All indices, parameters, and constraints are defined directly from the current CSV data.