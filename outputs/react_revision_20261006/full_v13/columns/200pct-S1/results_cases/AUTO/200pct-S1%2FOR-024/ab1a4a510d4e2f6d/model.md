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
   (Each musician/band $j$ receives exactly its demand $d_j$.)

2. **Warehouse activation:**  
   \[
   x_{ij} \leq M_j y_i, \quad \forall i \in I,\, \forall j \in J
   \]
   (No goods can be shipped from warehouse $i$ to musician/band $j$ unless warehouse $i$ is activated. $M_j$ is a valid upper bound for $d_j$.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Index Sets

- $I$: Set of warehouses (from fixed_cost.csv, column "Unnamed: 0")
- $J$: Set of musicians/bands (from demand.csv, column "customer")

##### Parameters and Data Mapping

- $d_j$: Demand of musician/band $j$ (from demand.csv, table_id: file_0_view_0, column: "demand", indexed by "customer")
- $f_i$: Fixed cost for warehouse $i$ (from fixed_cost.csv, table_id: file_1_view_0, column: "fixed_costs", indexed by "Unnamed: 0")
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to musician/band $j$ (from transportation_costs.csv, table_id: file_2_view_0, row: "Unnamed: 0", column: $j$)
- $M_j$: Set to $d_j$ (the demand of musician/band $j$), as this is the maximum that could be shipped to $j$ from any warehouse.

##### Data Mapping

- Warehouses $I$: file_1_view_0, column "Unnamed: 0"
- Musicians/Bands $J$: file_0_view_0, column "customer"
- Demand $d_j$: file_0_view_0, column "demand"
- Fixed cost $f_i$: file_1_view_0, column "fixed_costs"
- Transportation cost $c_{ij}$: file_2_view_0, row "Unnamed: 0" (warehouse), columns "C1", "C2", "C3" (musicians/bands)
- $M_j = d_j$ for each $j \in J$

No additional capacity limits are specified for warehouses.

##### Complete Model

\[
\begin{align*}
\min_{x_{ij},\, y_i} \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad
& \sum_{i \in I} x_{ij} = d_j, && \forall j \in J \\
& x_{ij} \leq d_j y_i, && \forall i \in I,\, \forall j \in J \\
& x_{ij} \geq 0, && \forall i \in I,\, \forall j \in J \\
& y_i \in \{0,1\}, && \forall i \in I
\end{align*}
\]

With all parameters and index sets mapped as above.