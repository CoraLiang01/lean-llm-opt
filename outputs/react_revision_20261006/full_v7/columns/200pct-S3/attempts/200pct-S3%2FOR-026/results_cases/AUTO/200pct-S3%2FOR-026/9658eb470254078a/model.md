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
2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from column `facility_id` in `cost.csv`)
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from column `customer_id` in `demand.csv`)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost for plant $i$  
  - Data: `fixed_opening_cost` column, `facility_id` key, table_id: `file_0_view_0`
- $u_i$: capacity of plant $i$  
  - Data: `facility_capacity` column, `facility_id` key, table_id: `file_0_view_0`
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$  
  - Data: `transportation_cost_to_Ck` columns, $k=1,\ldots,15$, `facility_id` key, table_id: `file_0_view_0`
- $d_j$: demand of customer $j$  
  - Data: `demand_units` column, `customer_id` key, table_id: `file_1_view_0`

##### Data Mapping

- Plants $I$:  
  - Table: `file_0_view_0`, column: `facility_id`
- Customers $J$:  
  - Table: `file_1_view_0`, column: `customer_id`
- Fixed opening cost $f_i$:  
  - Table: `file_0_view_0`, column: `fixed_opening_cost`, indexed by `facility_id`
- Plant capacity $u_i$:  
  - Table: `file_0_view_0`, column: `facility_capacity`, indexed by `facility_id`
- Transportation cost $c_{ij}$:  
  - Table: `file_0_view_0`, columns: `transportation_cost_to_C1` ... `transportation_cost_to_C15`, indexed by `facility_id` and customer $j$ (column suffix)
- Customer demand $d_j$:  
  - Table: `file_1_view_0`, column: `demand_units`, indexed by `customer_id`

No parameters are omitted or invented; all are mapped to the current CSV data.