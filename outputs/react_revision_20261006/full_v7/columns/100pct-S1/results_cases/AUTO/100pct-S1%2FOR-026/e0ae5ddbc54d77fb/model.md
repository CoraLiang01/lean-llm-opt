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
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants (facilities), from column `facility_id` in `file_0_view_0` (cost.csv)
- $J$: set of customers, from column `customer_id` in `file_1_view_0` (demand.csv)

##### Parameters and Data Mapping

- $f_i$: fixed opening cost of plant $i$, from column `fixed_opening_cost` in `file_0_view_0` (cost.csv)
- $\text{cap}_i$: capacity of plant $i$, from column `facility_capacity` in `file_0_view_0` (cost.csv)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$, from columns `transportation_cost_to_C1` ... `transportation_cost_to_C15` in `file_0_view_0` (cost.csv), with $j$ matching customer index.
- $d_j$: demand of customer $j$, from column `demand_units` in `file_1_view_0` (demand.csv)

##### Data Mapping

- Plants: $I$ = all `facility_id` in `file_0_view_0` (cost.csv)
- Customers: $J$ = all `customer_id` in `file_1_view_0` (demand.csv)
- $f_i$: `fixed_opening_cost` in `file_0_view_0` (cost.csv), indexed by `facility_id`
- $\text{cap}_i$: `facility_capacity` in `file_0_view_0` (cost.csv), indexed by `facility_id`
- $c_{ij}$: `transportation_cost_to_Ck` in `file_0_view_0` (cost.csv), where $k$ matches $j$ in $J$
- $d_j$: `demand_units` in `file_1_view_0` (demand.csv), indexed by `customer_id`