##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i$ to customer $j$ (continuous), for all $i \in I$, $j \in J$.
- $y_i \in \{0,1\}$: 1 if plant $i$ is built, 0 otherwise, for all $i \in I$.

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
   \sum_{j \in J} x_{ij} \leq \text{cap}_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of plants, from column `facility_id` in `cost.csv` (`file_0_view_0`)
- $J$: set of customers, from column `customer_id` in `demand.csv` (`file_1_view_0`)
- $f_i$: fixed opening cost of plant $i$, from column `fixed_opening_cost` in `cost.csv` (`file_0_view_0`)
- $\text{cap}_i$: capacity of plant $i$, from column `facility_capacity` in `cost.csv` (`file_0_view_0`)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from columns `transportation_cost_to_Cj` in `cost.csv` (`file_0_view_0`)
- $d_j$: demand of customer $j$, from column `demand_units` in `demand.csv` (`file_1_view_0`)

##### Data Mapping

- Plants $I$: `facility_id` in `cost.csv` (`file_0_view_0`)
- Customers $J$: `customer_id` in `demand.csv` (`file_1_view_0`)
- Fixed opening cost $f_i$: `fixed_opening_cost` in `cost.csv` (`file_0_view_0`)
- Plant capacity $\text{cap}_i$: `facility_capacity` in `cost.csv` (`file_0_view_0`)
- Transportation cost $c_{ij}$: `transportation_cost_to_Cj` in `cost.csv` (`file_0_view_0`), where $j$ matches customer $Cj$
- Demand $d_j$: `demand_units` in `demand.csv` (`file_1_view_0`)

All index sets and parameters are defined by the full set of rows and columns in the respective files as described above.