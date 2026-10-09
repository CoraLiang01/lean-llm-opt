##### Decision Variables

$x_{ij} \geq 0$: quantity of Adidas products shipped from supplier $i \in I$ to store $j \in J$ (continuous).

$y_i \in \{0,1\}$: whether supplier $i$ is operational (binary).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i$

##### Constraints

1. Store demand satisfaction:
   $$
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   $$

2. Supplier activation logic:
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   $$
   where $M_i = \sum_{j \in J} d_j$ is a valid upper bound for each supplier (since there are no explicit supplier capacity limits).

3. Variable domains:
   $$
   x_{ij} \geq 0, \quad y_i \in \{0,1\}, \quad \forall i \in I,\, j \in J
   $$

##### Index Sets

$I = \{$S1, S2, S3, S4, S5, S6$\}$ (suppliers, from fixed_cost.csv and transportation_costs.csv Unnamed: 0)

$J = \{$C1, C2, C3, C4, C5, C6$\}$ (stores, from demand.csv customer and transportation_costs.csv columns)

##### Parameter Mapping

- $d_j$: demand for store $j$ (from demand.csv, table_id: file_0_view_0, column: demand, index: customer)
- $f_i$: fixed cost for supplier $i$ (from fixed_cost.csv, table_id: file_1_view_0, column: fixed_costs, index: Unnamed: 0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, row: Unnamed: 0, column: $j$)
- $M_i$: $\sum_{j \in J} d_j$ (sum over all store demands, from demand.csv)

##### Data Mapping

- Store demand: demand.csv (table_id: file_0_view_0, columns: customer, demand)
- Supplier fixed cost: fixed_cost.csv (table_id: file_1_view_0, columns: Unnamed: 0, fixed_costs)
- Transportation cost matrix: transportation_costs.csv (table_id: file_2_view_0, rows: Unnamed: 0, columns: C1–C6)

All index sets, parameters, and constraints are mapped directly to the provided CSV data. No additional capacity or conditional bounds are imposed beyond those described.