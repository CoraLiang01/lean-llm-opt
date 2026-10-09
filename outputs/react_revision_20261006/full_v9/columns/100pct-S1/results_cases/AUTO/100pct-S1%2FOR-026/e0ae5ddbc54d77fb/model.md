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
   \sum_{j \in J} x_{ij} \leq s_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants (facilities), from column `facility_id` in `cost.csv` (`file_0_view_0`)
- $J$: set of customers, from column `customer_id` in `demand.csv` (`file_1_view_0`)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost for plant $i$, from column `fixed_opening_cost` in `cost.csv` (`file_0_view_0`)
- $s_i$: capacity of plant $i$, from column `facility_capacity` in `cost.csv` (`file_0_view_0`)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from columns `transportation_cost_to_C1` ... `transportation_cost_to_C15` in `cost.csv` (`file_0_view_0`)
- $d_j$: demand of customer $j$, from column `demand_units` in `demand.csv` (`file_1_view_0`)

##### Data Mapping

- Plants $I$: `facility_id` in `cost.csv` (`file_0_view_0`)
- Customers $J$: `customer_id` in `demand.csv` (`file_1_view_0`)
- $f_i$: `fixed_opening_cost` in `cost.csv` (`file_0_view_0`)
- $s_i$: `facility_capacity` in `cost.csv` (`file_0_view_0`)
- $c_{ij}$: `transportation_cost_to_Ck` in `cost.csv` (`file_0_view_0`), where $j$ corresponds to customer $Ck$
- $d_j$: `demand_units` in `demand.csv` (`file_1_view_0`)

No parameters are omitted or abbreviated; all index sets and coefficients are mapped directly to the CSV columns as described.