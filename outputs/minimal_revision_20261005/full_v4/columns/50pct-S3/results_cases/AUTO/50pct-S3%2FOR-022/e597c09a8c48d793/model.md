##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I$: Set of suppliers, from column facility_id in table_id file_1_view_0.
- $J$: Set of branches, from column customer_id in table_id file_0_view_0.
- $d_j$: Demand of branch $j$, from column demand_units in table_id file_0_view_0.
- $f_i$: Fixed opening cost for supplier $i$, from column fixed_opening_cost in table_id file_1_view_0.
- $c_{ij}$: Transportation cost per unit from supplier $i$ to branch $j$, from column transportation_cost_to_{j} in table_id file_2_view_0, with $i$ from facility_id and $j$ from customer_id.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Branch demand satisfaction**:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation logic** (inactive suppliers cannot ship):
   \[
   \sum_{j \in J} x_{ij} \leq M \cdot y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j$ (total demand), computed from demand_units in file_0_view_0.

3. **Variable domains**:
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: facility_id from table_id file_1_view_0
- $J$: customer_id from table_id file_0_view_0
- $d_j$: demand_units from table_id file_0_view_0, column demand_units, indexed by customer_id
- $f_i$: fixed_opening_cost from table_id file_1_view_0, column fixed_opening_cost, indexed by facility_id
- $c_{ij}$: transportation_cost_to_{j} from table_id file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C5, indexed by facility_id and customer_id
- $M$: $\sum_{j \in J} d_j$, sum over demand_units in file_0_view_0

All index sets and parameters are defined exactly as present in the source tables.