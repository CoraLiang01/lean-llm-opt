##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly their demand $d_j$.)

2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i, \quad \forall i \in I
   \]
   (No goods can be shipped from warehouse $i$ unless it is activated. $M_i$ is a sufficiently large upper bound, e.g., $M_i = \sum_{j \in J} d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of warehouses (from fixed_cost.csv, column "Unnamed: 0")
- $J$: Set of musicians/bands (from demand.csv, column "customer")

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$  
  — Source: demand.csv, table_id: file_0_view_0, column: "demand", indexed by "customer"
- $f_i$: Fixed cost for warehouse $i$  
  — Source: fixed_cost.csv, table_id: file_1_view_0, column: "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$  
  — Source: transportation_costs.csv, table_id: file_2_view_0, row: "Unnamed: 0" (warehouses), columns: "C1", "C2", "C3" (musicians/bands)
- $M_i$: Big-M parameter for each warehouse $i$  
  — $M_i = \sum_{j \in J} d_j$ (sum of all demands, computed from demand.csv)

##### Data Mapping

- $I = \{$ values in fixed_cost.csv, table_id: file_1_view_0, column "Unnamed: 0" $\}$
- $J = \{$ values in demand.csv, table_id: file_0_view_0, column "customer" $\}$
- $d_j$: demand.csv, table_id: file_0_view_0, column "demand", indexed by "customer"
- $f_i$: fixed_cost.csv, table_id: file_1_view_0, column "fixed_costs", indexed by "Unnamed: 0"
- $c_{ij}$: transportation_costs.csv, table_id: file_2_view_0, row "Unnamed: 0", columns "C1", "C2", "C3"
- $M_i = \sum_{j \in J} d_j$ (from demand.csv, table_id: file_0_view_0, column "demand")

No additional constraints or capacity limits are specified beyond the above. All index sets and parameters are defined directly from the CSV data.