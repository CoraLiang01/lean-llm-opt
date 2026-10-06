##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from file_1_view_0.facility_id and file_2_view_0.facility_id.
- $J$: Set of branches, from file_0_view_0.customer_id and file_2_view_0 columns transportation_cost_to_C1, ..., transportation_cost_to_C5.
- $d_j$: Demand at branch $j$, from file_0_view_0.demand_units.
- $f_i$: Fixed opening cost for supplier $i$, from file_1_view_0.fixed_opening_cost.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from file_2_view_0, column transportation_cost_to_{j} for row facility_id $i$.
- $M$: A sufficiently large constant, $M = \sum_{j \in J} d_j$.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction at each branch:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\}, \quad \forall i \in I
   \]

##### Data Mapping

- $I$: All facility_id in file_1_view_0 and file_2_view_0.
- $J$: All customer_id in file_0_view_0 and all columns transportation_cost_to_{customer_id} in file_2_view_0.
- $d_j$: file_0_view_0.demand_units, indexed by customer_id.
- $f_i$: file_1_view_0.fixed_opening_cost, indexed by facility_id.
- $c_{ij}$: file_2_view_0, value in column transportation_cost_to_{j} for row facility_id $i$.
- $M$: $\sum_{j \in J} d_j$ (sum over file_0_view_0.demand_units).

All index sets and parameters are defined by the full set of entities present in the respective columns of the provided CSV files.