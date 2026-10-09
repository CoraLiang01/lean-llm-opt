##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
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

- $I$: Set of warehouses (from fixed_cost.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J$: Set of musicians/bands (from demand.csv, column "customer", table_id: file_0_view_0)

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$ (from demand.csv, column "demand", table_id: file_0_view_0)
- $f_i$: Fixed cost for activating warehouse $i$ (from fixed_cost.csv, column "fixed_costs", table_id: file_1_view_0)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from transportation_costs.csv, row "Unnamed: 0" for $i$, column $j$, table_id: file_2_view_0)
- $M_i$: Big-M parameter for each warehouse $i$ (set as $M_i = \sum_{j \in J} d_j$)

##### Data Mapping

- $I$: file_1_view_0, column "Unnamed: 0"
- $J$: file_0_view_0, column "customer"
- $d_j$: file_0_view_0, columns "customer", "demand"
- $f_i$: file_1_view_0, columns "Unnamed: 0", "fixed_costs"
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" (warehouse $i$), columns "C1", "C2", "C3" (musician/band $j$)
- $M_i$: $M_i = \sum_{j \in J} d_j$ (sum over all $d_j$ from file_0_view_0)

No additional capacity or proportion constraints are specified beyond the above.