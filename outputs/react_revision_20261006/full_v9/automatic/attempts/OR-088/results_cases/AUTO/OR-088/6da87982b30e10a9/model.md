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

- $I$: set of plants, $I = \{$plant$\}$ from [file_0_view_0, column: plant]
- $J$: set of customers, $J = \{$customer$\}$ from [file_1_view_0, column: customer]

##### Parameters and Data Mapping

- $f_i$: fixed opening cost of plant $i$  
  [file_0_view_0, columns: plant, fixed_cost]
- $\text{cap}_i$: capacity of plant $i$  
  [file_0_view_0, columns: plant, capacity]
- $c_{ij}$: per-unit transport cost from plant $i$ to customer $j$  
  [file_0_view_0, row: plant $i$, column: $j$ (C1–C15)]
- $d_j$: demand of customer $j$  
  [file_1_view_0, columns: customer, demand]

##### Data Mapping

- Plants $I$: [file_0_view_0, column: plant]
- Customers $J$: [file_1_view_0, column: customer]
- Fixed costs $f_i$: [file_0_view_0, columns: plant, fixed_cost]
- Capacities $\text{cap}_i$: [file_0_view_0, columns: plant, capacity]
- Transport costs $c_{ij}$: [file_0_view_0, row: plant $i$, columns: C1–C15]
- Demands $d_j$: [file_1_view_0, columns: customer, demand]