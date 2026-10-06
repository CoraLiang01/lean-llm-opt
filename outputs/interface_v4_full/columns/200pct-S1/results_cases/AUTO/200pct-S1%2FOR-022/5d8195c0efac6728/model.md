##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (set of suppliers/facilities)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$ (set of branches/customers)

- Demand for each branch:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$

- Fixed opening cost for each supplier:
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$

- Transportation cost per unit from supplier $i$ to branch $j$ ($c_{ij}$):

|        | C1      | C2      | C3     | C4      | C5     |
|--------|---------|---------|--------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (sufficiently large upper bound for each supplier's total shipment)

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction for each branch:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation constraint:**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Full Model with Data

\[
\begin{align*}
\min \quad & 
\Big[
150.74\,x_{\text{S1},\text{C1}} + 0.02\,x_{\text{S1},\text{C2}} + 49.13\,x_{\text{S1},\text{C3}} + 2080.15\,x_{\text{S1},\text{C4}} + 426.4\,x_{\text{S1},\text{C5}} \\
& + 233.05\,x_{\text{S2},\text{C1}} + 97.73\,x_{\text{S2},\text{C2}} + 49.84\,x_{\text{S2},\text{C3}} + 1982.39\,x_{\text{S2},\text{C4}} + 23.96\,x_{\text{S2},\text{C5}} \\
& + 55.68\,x_{\text{S3},\text{C1}} + 935.61\,x_{\text{S3},\text{C2}} + 4.03\,x_{\text{S3},\text{C3}} + 73.09\,x_{\text{S3},\text{C4}} + 525.32\,x_{\text{S3},\text{C5}} \\
& + 1483.82\,x_{\text{S4},\text{C1}} + 1801.08\,x_{\text{S4},\text{C2}} + 112.16\,x_{\text{S4},\text{C3}} + 816.05\,x_{\text{S4},\text{C4}} + 107.01\,x_{\text{S4},\text{C5}} \\
& + 1119.47\,x_{\text{S5},\text{C1}} + 884.31\,x_{\text{S5},\text{C2}} + 0.08\,x_{\text{S5},\text{C3}} + 1544.95\,x_{\text{S5},\text{C4}} + 543.67\,x_{\text{S5},\text{C5}} \\
& + 97.65\,y_{\text{S1}} + 99.76\,y_{\text{S2}} + 100.76\,y_{\text{S3}} + 105.32\,y_{\text{S4}} + 98.88\,y_{\text{S5}}
\Big]
\end{align*}
\]

Subject to:

\[
\begin{align*}
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} + x_{\text{S5},\text{C1}} = 143 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} + x_{\text{S5},\text{C2}} = 6 \\
& x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} + x_{\text{S5},\text{C3}} = 10 \\
& x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} + x_{\text{S5},\text{C4}} = 25 \\
& x_{\text{S1},\text{C5}} + x_{\text{S2},\text{C5}} + x_{\text{S3},\text{C5}} + x_{\text{S4},\text{C5}} + x_{\text{S5},\text{C5}} = 3 \\
\\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} + x_{\text{S1},\text{C5}} \leq 187\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} + x_{\text{S2},\text{C5}} \leq 187\,y_{\text{S2}} \\
& x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} + x_{\text{S3},\text{C5}} \leq 187\,y_{\text{S3}} \\
& x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} + x_{\text{S4},\text{C5}} \leq 187\,y_{\text{S4}} \\
& x_{\text{S5},\text{C1}} + x_{\text{S5},\text{C2}} + x_{\text{S5},\text{C3}} + x_{\text{S5},\text{C4}} + x_{\text{S5},\text{C5}} \leq 187\,y_{\text{S5}} \\
\\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

##### All required parameters (vectors and matrices):

- Facilities: $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Branches: $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$
- Demand vector: $d = [143,\, 6,\, 10,\, 25,\, 3]$ (ordered as C1, C2, C3, C4, C5)
- Fixed cost vector: $f = [97.65,\, 99.76,\, 100.76,\, 105.32,\, 98.88]$ (ordered as S1, S2, S3, S4, S5)
- Transportation cost matrix $C = [c_{ij}]$ (rows: S1–S5, columns: C1–C5):

\[
C = \begin{bmatrix}
150.74 & 0.02 & 49.13 & 2080.15 & 426.4 \\
233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
\end{bmatrix}
\]

- $M = 187$

This model determines which suppliers to open and how much each should supply to each branch, minimizing the total of fixed and transportation costs, while meeting all branch demands.