##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each customer’s demand must be fully met.)

2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
   (A plant can only supply up to its capacity if it is built; if not built, it supplies nothing.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$, from column `facility_id` in `file_0_view_0` (`cost.csv`).
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$, from column `customer_id` in `file_1_view_0` (`demand.csv`).

##### Parameters and Data Mapping

- $f_i$: fixed opening cost for plant $i$  
  — `file_0_view_0`, columns:  
    - plant index: `facility_id`  
    - value: `fixed_opening_cost`

- $\text{cap}_i$: capacity of plant $i$  
  — `file_0_view_0`, columns:  
    - plant index: `facility_id`  
    - value: `facility_capacity`

- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$  
  — `file_0_view_0`, columns:  
    - plant index: `facility_id`  
    - customer index: for customer $j$, use column `transportation_cost_to_{j}` (e.g., `transportation_cost_to_C1`, ..., `transportation_cost_to_C15`)

- $d_j$: demand of customer $j$  
  — `file_1_view_0`, columns:  
    - customer index: `customer_id`  
    - value: `demand_units`

##### Data Mapping

- Plants ($I$):  
  - Table: `file_0_view_0` (`cost.csv`), column: `facility_id`
- Customers ($J$):  
  - Table: `file_1_view_0` (`demand.csv`), column: `customer_id`
- Fixed opening cost ($f_i$):  
  - Table: `file_0_view_0`, column: `fixed_opening_cost`
- Plant capacity ($\text{cap}_i$):  
  - Table: `file_0_view_0`, column: `facility_capacity`
- Transportation cost ($c_{ij}$):  
  - Table: `file_0_view_0`, columns: `transportation_cost_to_C1`, ..., `transportation_cost_to_C15`
- Customer demand ($d_j$):  
  - Table: `file_1_view_0`, column: `demand_units`