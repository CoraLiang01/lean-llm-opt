##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}\}$: Set of warehouses (facilities).
- $J = \{\text{C1}, \text{C2}, \text{C3}\}$: Set of musicians/bands (customers).
- $d_j$: Demand of customer $j$.
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
- $f_i$: Fixed cost of opening warehouse $i$.
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$.

|           | C1      | C2      | C3      |
|-----------|---------|---------|---------|
| **S1**    | 1506.22 | 70.9    | 8.44    |
| **S2**    | 1732.65 | 1780.72 | 567.44  |
| **S3**    | 115.66  | 100.76  | 64.68   |

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$ (sufficiently large upper bound for linking constraints).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model with Data

\[
\begin{align*}
\min\quad & 1506.22\,x_{\text{S1},\text{C1}} + 70.9\,x_{\text{S1},\text{C2}} + 8.44\,x_{\text{S1},\text{C3}} \\
& + 1732.65\,x_{\text{S2},\text{C1}} + 1780.72\,x_{\text{S2},\text{C2}} + 567.44\,x_{\text{S2},\text{C3}} \\
& + 115.66\,x_{\text{S3},\text{C1}} + 100.76\,x_{\text{S3},\text{C2}} + 64.68\,x_{\text{S3},\text{C3}} \\
& + 102.33\,y_{\text{S1}} + 94.92\,y_{\text{S2}} + 91.83\,y_{\text{S3}}
\end{align*}
\]

Subject to:

\[
\begin{align*}
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} = 1083 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} = 776 \\
& x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} = 16214 \\
\\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} \leq 18073\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} \leq 18073\,y_{\text{S2}} \\
& x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} \leq 18073\,y_{\text{S3}} \\
\\
& x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\text{S2},\text{S3}\},\, j \in \{\text{C1},\text{C2},\text{C3}\} \\
& y_i \in \{0,1\} \quad \forall i \in \{\text{S1},\text{S2},\text{S3}\}
\end{align*}
\]

##### Retrieved Information

{
  "warehouses": ["S1", "S2", "S3"],
  "musicians_bands": ["C1", "C2", "C3"],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "cost": {
    "S1": {"C1": 1506.22, "C2": 70.9, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  },
  "M": 18073
}