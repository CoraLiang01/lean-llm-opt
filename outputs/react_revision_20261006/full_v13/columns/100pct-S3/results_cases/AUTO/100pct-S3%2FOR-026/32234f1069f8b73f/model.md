##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from facility $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether facility $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. **Facility capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of facilities (plants), from column `facility_id` in `cost.csv` ([table_id: file_0_view_0]).
- $J$: set of customers, from column `customer_id` in `demand.csv` ([table_id: file_1_view_0]).
- $f_i$: fixed opening cost of facility $i$, from column `fixed_opening_cost` in `cost.csv` ([table_id: file_0_view_0]).
- $K_i$: capacity of facility $i$, from column `facility_capacity` in `cost.csv` ([table_id: file_0_view_0]).
- $c_{ij}$: per-unit transportation cost from facility $i$ to customer $j$, from columns `transportation_cost_to_C1` ... `transportation_cost_to_C15` in `cost.csv` ([table_id: file_0_view_0]), with $j$ matching customer $j$.
- $d_j$: demand of customer $j$, from column `demand_units` in `demand.csv` ([table_id: file_1_view_0]).

##### Data Mapping

- Facilities $I$: `facility_id` in [file_0_view_0]
- Customers $J$: `customer_id` in [file_1_view_0]
- Fixed opening cost $f_i$: `fixed_opening_cost` in [file_0_view_0]
- Facility capacity $K_i$: `facility_capacity` in [file_0_view_0]
- Transportation cost $c_{ij}$: `transportation_cost_to_Ck` in [file_0_view_0], where $j$ corresponds to customer $Ck$
- Demand $d_j$: `demand_units` in [file_1_view_0]

No parameters are omitted or abbreviated; all are mapped directly to the source columns and index sets.