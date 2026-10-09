##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   (Each musician/band $j$ receives exactly their demand $d_j$.)

2. **Warehouse activation:**  
   \[
   x_{ij} \leq d_j y_i, \quad \forall i \in I,\, \forall j \in J
   \]
   (No goods can be shipped from inactive warehouses.)

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \text{ continuous}, \quad y_i \in \{0,1\}
   \]

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}\}$ (warehouses, from fixed_cost.csv and transportation_costs.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}\}$ (musicians/bands, from demand.csv and transportation_costs.csv)
- $d_j$: demand for each $j \in J$, from demand.csv:
  - C1: 1083
  - C2: 776
  - C3: 16214
  - C4: 553
  - C5: 17106
  - C6: 594
  - C7: 732
- $f_i$: fixed cost for each $i \in I$, from fixed_cost.csv:
  - S1: 102.33
  - S2: 94.92
  - S3: 91.83
  - S4: 98.71
  - S5: 95.73
  - S6: 99.96
  - S7: 98.16
- $c_{ij}$: transportation cost per unit from warehouse $i$ to musician/band $j$, from transportation_costs.csv (matrix, rows $i$ in $I$, columns $j$ in $J$).

**Source-column Data Mapping:**  
- demand.csv: customer $\to$ $j$, demand $\to$ $d_j$  
- fixed_cost.csv: Unnamed: 0 $\to$ $i$, fixed_costs $\to$ $f_i$  
- transportation_costs.csv: Unnamed: 0 $\to$ $i$, $Ck$ columns $\to$ $c_{ij}$ for $j = Ck$