##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise.

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: per-unit transportation cost from plant $i$ to customer $j$
- $f_i$: fixed opening cost for plant $i$

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   Each customer's demand must be fully met.

2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq \text{cap}_i \, y_i, \quad \forall i \in I
   \]
   No plant can supply more than its capacity, and only if it is opened.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: set of plants, as given by the "plant" column in `cost.csv`
- $J$: set of customers, as given by the "customer" column in `demand.csv`

##### Data Mapping

- $f_i$: `cost.csv`, column `fixed_cost`, indexed by `plant` (table_id: file_0_view_0, column: fixed_cost)
- $\text{cap}_i$: `cost.csv`, column `capacity`, indexed by `plant` (table_id: file_0_view_0, column: capacity)
- $c_{ij}$: `cost.csv`, columns `C1`–`C15`, indexed by `plant` and customer (table_id: file_0_view_0, columns: C1–C15)
- $d_j$: `demand.csv`, column `demand`, indexed by `customer` (table_id: file_1_view_0, column: demand)
- $I$: all `plant` values in `cost.csv` (table_id: file_0_view_0, column: plant)
- $J$: all `customer` values in `demand.csv` (table_id: file_1_view_0, column: customer)

All parameters and sets are defined exactly as in the supplied CSV data.