##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether plant $i$ is built (opened).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Customer demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Plant capacity (only if built):**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Index Sets and Parameters

- $I$: set of plants, $I = \{\text{F1}, \ldots, \text{F15}\}$ (from file_0_view_0, column facility_id)
- $J$: set of customers, $J = \{\text{C1}, \ldots, \text{C15}\}$ (from file_1_view_0, column customer_id)
- $f_i$: fixed opening cost for plant $i$ (from file_0_view_0, column fixed_opening_cost)
- $K_i$: capacity of plant $i$ (from file_0_view_0, column facility_capacity)
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$ (from file_0_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C15)
- $d_j$: demand of customer $j$ (from file_1_view_0, column demand_units)

##### Data Mapping

- Plants $I$ and their parameters $f_i$, $K_i$, and $c_{ij}$:  
  - Table: file_0_view_0 (cost.csv)
    - Plant index: facility_id
    - Fixed cost: fixed_opening_cost
    - Capacity: facility_capacity
    - Transportation cost: transportation_cost_to_C1, ..., transportation_cost_to_C15 (columns indexed by customer $j$)
- Customers $J$ and their demands $d_j$:
  - Table: file_1_view_0 (demand.csv)
    - Customer index: customer_id
    - Demand: demand_units

All index sets, parameters, and constraints are defined directly from the CSV data as described above.