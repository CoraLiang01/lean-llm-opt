##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if plant $i$ is built (opened), 0 otherwise.

##### Parameters

- $I$: set of plants (from `cost.csv`, column `plant`, table_id: file_0_view_0)
- $J$: set of customers (from `demand.csv`, column `customer`, table_id: file_1_view_0)
- $f_i$: fixed opening cost for plant $i$ (from `cost.csv`, column `fixed_cost`, table_id: file_0_view_0)
- $K_i$: capacity of plant $i$ (from `cost.csv`, column `capacity`, table_id: file_0_view_0)
- $d_j$: demand of customer $j$ (from `demand.csv`, column `demand`, table_id: file_1_view_0)
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$ (from `cost.csv`, columns `C1`–`C15`, table_id: file_0_view_0, with row_id_mapping: plant, column_id_mapping: customer)

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Plant capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I$: `cost.csv`, table_id: file_0_view_0, column: `plant`
- $J$: `demand.csv`, table_id: file_1_view_0, column: `customer`
- $f_i$: `cost.csv`, table_id: file_0_view_0, column: `fixed_cost`, row_id_mapping: `plant`
- $K_i$: `cost.csv`, table_id: file_0_view_0, column: `capacity`, row_id_mapping: `plant`
- $d_j$: `demand.csv`, table_id: file_1_view_0, column: `demand`, row_id_mapping: `customer`
- $c_{ij}$: `cost.csv`, table_id: file_0_view_0, columns: `C1`–`C15`, row_id_mapping: `plant`, column_id_mapping: `customer`