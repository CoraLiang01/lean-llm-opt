##### Decision Variables

- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise, for all $i \in I$
- $x_{ij} \geq 0$: quantity shipped from factory $i$ to distribution center $j$, for all $i \in I$, $j \in J$

##### Parameters

- Factories $I = \{A1, A2, ..., A15\}$
- Distribution centers $J = \{B1, B2, ..., B8\}$

- Fixed costs and capacities (from facility_costs.csv):

| Factory | Fixed Cost $f_i$ | Capacity $K_i$ |
|---------|------------------|---------------|
| A1      | 0                | 30            |
| A2      | 175              | 10            |
| A3      | 300              | 20            |
| A4      | 375              | 30            |
| A5      | 500              | 40            |
| A6      | 200              | 20            |
| A7      | 260              | 25            |
| A8      | 220              | 30            |
| A9      | 320              | 35            |
| A10     | 280              | 20            |
| A11     | 350              | 40            |
| A12     | 420              | 25            |
| A13     | 470              | 30            |
| A14     | 520              | 50            |
| A15     | 560              | 45            |

- Demands (from demand_requirements.csv):

| Distribution Center | Demand $d_j$ |
|---------------------|--------------|
| B1                  | 30           |
| B2                  | 25           |
| B3                  | 20           |
| B4                  | 35           |
| B5                  | 25           |
| B6                  | 30           |
| B7                  | 25           |
| B8                  | 30           |

- Shipping costs $c_{ij}$ (from shipping_costs.csv):

| Factory | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|---------|----|----|----|----|----|----|----|----|
| A1      | 8  | 4  | 3  | 6  | 7  | 5  | 9  | 8  |
| A2      | 5  | 2  | 3  | 5  | 6  | 4  | 7  | 6  |
| A3      | 4  | 3  | 4  | 6  | 5  | 5  | 6  | 7  |
| A4      | 9  | 7  | 5  | 8  | 9  | 6  | 10 | 7  |
| A5      | 10 | 4  | 2  | 6  | 8  | 5  | 7  | 3  |
| A6      | 6  | 5  | 4  | 5  | 7  | 6  | 8  | 5  |
| A7      | 7  | 6  | 5  | 4  | 6  | 7  | 9  | 6  |
| A8      | 5  | 4  | 6  | 3  | 5  | 6  | 7  | 6  |
| A9      | 8  | 7  | 6  | 7  | 9  | 8  | 10 | 7  |
| A10     | 6  | 5  | 7  | 4  | 6  | 5  | 7  | 5  |
| A11     | 9  | 6  | 4  | 6  | 8  | 7  | 9  | 6  |
| A12     | 7  | 5  | 6  | 5  | 6  | 5  | 8  | 5  |
| A13     | 8  | 6  | 5  | 6  | 7  | 6  | 8  | 7  |
| A14     | 9  | 5  | 3  | 5  | 7  | 4  | 6  | 4  |
| A15     | 10 | 6  | 4  | 5  | 8  | 5  | 7  | 5  |

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]

2. **Factory capacity (only if opened):**
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i \qquad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\min\quad & \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} \\
\text{s.t.}\quad & \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq K_i y_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I \\
\end{align*}
\]

Where all parameters ($f_i$, $K_i$, $d_j$, $c_{ij}$) are as listed above.