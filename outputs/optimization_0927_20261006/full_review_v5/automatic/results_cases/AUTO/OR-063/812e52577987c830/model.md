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
2. **Warehouse activation:**  
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 37,\!058$ is a valid upper bound.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}\}$ (warehouses)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}\}$ (musicians/bands)

###### Demand

| Customer | Demand |
|----------|--------|
| C1       | 1083   |
| C2       | 776    |
| C3       | 16214  |
| C4       | 553    |
| C5       | 17106  |
| C6       | 594    |
| C7       | 732    |

###### Warehouse Fixed Costs

| Warehouse | Fixed Cost |
|-----------|------------|
| S1        | 102.33     |
| S2        | 94.92      |
| S3        | 91.83      |
| S4        | 98.71      |
| S5        | 95.73      |
| S6        | 99.96      |
| S7        | 98.16      |

###### Transportation Costs ($c_{ij}$)

| Warehouse |   C1    |   C2    |   C3    |   C4    |   C5    |   C6    |   C7    |
|-----------|---------|---------|---------|---------|---------|---------|---------|
| S1        | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2        | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3        | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4        | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5        | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6        | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7        | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

##### Parameter $M$

\[
M = \sum_{j \in J} d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 37,\!058
\]

##### Model Summary

- Decision variables: $x_{ij}$ (continuous, $\geq 0$), $y_i$ (binary)
- Objective: Minimize total transportation and warehouse activation costs
- Each musician/band's demand must be fully met
- Warehouses can only ship if activated; no explicit warehouse capacity limits

All parameters and data are as retrieved from the CSV files.