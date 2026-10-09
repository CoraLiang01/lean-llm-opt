##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (Suppliers)
- $J = \{C1, C2, C3, C4, C5\}$ (Branches)

- Demands:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed costs:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation costs $c_{ij}$ (per unit from supplier $i$ to branch $j$):

|        | C1      | C2      | C3     | C4      | C5      |
|--------|---------|---------|--------|---------|---------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.40  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96   |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32  |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01  |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67  |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction (each branch receives exactly its demand):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (inactive suppliers cannot ship):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model (with all parameters)

\[
\begin{align*}
\min \quad & 
\Big[ 
150.74\,x_{S1,C1} + 0.02\,x_{S1,C2} + 49.13\,x_{S1,C3} + 2080.15\,x_{S1,C4} + 426.4\,x_{S1,C5} \\
&+ 233.05\,x_{S2,C1} + 97.73\,x_{S2,C2} + 49.84\,x_{S2,C3} + 1982.39\,x_{S2,C4} + 23.96\,x_{S2,C5} \\
&+ 55.68\,x_{S3,C1} + 935.61\,x_{S3,C2} + 4.03\,x_{S3,C3} + 73.09\,x_{S3,C4} + 525.32\,x_{S3,C5} \\
&+ 1483.82\,x_{S4,C1} + 1801.08\,x_{S4,C2} + 112.16\,x_{S4,C3} + 816.05\,x_{S4,C4} + 107.01\,x_{S4,C5} \\
&+ 1119.47\,x_{S5,C1} + 884.31\,x_{S5,C2} + 0.08\,x_{S5,C3} + 1544.95\,x_{S5,C4} + 543.67\,x_{S5,C5} \\
&+ 97.65\,y_{S1} + 99.76\,y_{S2} + 100.76\,y_{S3} + 105.32\,y_{S4} + 98.88\,y_{S5}
\Big]
\end{align*}
\]

Subject to:

\[
\begin{align*}
&x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} + x_{S5,C1} = 143 \\
&x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} + x_{S5,C2} = 6 \\
&x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} + x_{S5,C3} = 10 \\
&x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} + x_{S5,C4} = 25 \\
&x_{S1,C5} + x_{S2,C5} + x_{S3,C5} + x_{S4,C5} + x_{S5,C5} = 3 \\
\\
&x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} + x_{S1,C5} \leq 187\,y_{S1} \\
&x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} + x_{S2,C5} \leq 187\,y_{S2} \\
&x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} + x_{S3,C5} \leq 187\,y_{S3} \\
&x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} + x_{S4,C5} \leq 187\,y_{S4} \\
&x_{S5,C1} + x_{S5,C2} + x_{S5,C3} + x_{S5,C4} + x_{S5,C5} \leq 187\,y_{S5} \\
\\
&x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
&y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]