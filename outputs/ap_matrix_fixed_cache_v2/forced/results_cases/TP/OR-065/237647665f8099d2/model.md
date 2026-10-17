##### Sets

- Warehouses (facilities): $I = \{S1, S2, S3\}$
- Musicians/bands (customers): $J = \{C1, C2, C3\}$

##### Parameters

- Demand for each customer:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
- Fixed cost for each warehouse:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
- Transportation cost per unit from warehouse $i$ to customer $j$ ($c_{ij}$):

\[
\begin{array}{c|ccc}
 & C1 & C2 & C3 \\
\hline
S1 & 1506.22 & 70.90 & 8.44 \\
S2 & 1732.65 & 1780.72 & 567.44 \\
S3 & 115.66 & 100.76 & 64.68 \\
\end{array}
\]

##### Decision Variables

- $y_i \in \{0,1\}$: $1$ if warehouse $i$ is open, $0$ otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to customer $j$, for $i \in I$, $j \in J$

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

That is,

\[
\min\ \ 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} + 1506.22\,x_{S1,C1} + 70.90\,x_{S1,C2} + 8.44\,x_{S1,C3} + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3}
\]

##### Constraints

1. **Demand satisfaction:** For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   That is,
   \begin{align*}
   x_{S1,C1} + x_{S2,C1} + x_{S3,C1} &= 1083 \\
   x_{S1,C2} + x_{S2,C2} + x_{S3,C2} &= 776 \\
   x_{S1,C3} + x_{S2,C3} + x_{S3,C3} &= 16214 \\
   \end{align*}

2. **Facility activation:** For all $i \in I$, $j \in J$,
   \[
   x_{ij} \leq d_j y_i
   \]
   (A warehouse can only supply to a customer if it is open; $d_j$ is a valid upper bound.)

3. **Variable domains:**
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Complete Model

\[
\begin{align*}
\min\ & 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\
&+ 1506.22\,x_{S1,C1} + 70.90\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
&+ 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
&+ 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\
& x_{ij} \leq d_j y_i \quad \forall i \in \{S1,S2,S3\},\ j \in \{C1,C2,C3\} \\
& y_i \in \{0,1\} \quad \forall i \in \{S1,S2,S3\} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1,S2,S3\},\ j \in \{C1,C2,C3\}
\end{align*}
\]

###### Retrieved Information

{
  "customers": [
    {"customer": "C1", "demand": 1083},
    {"customer": "C2", "demand": 776},
    {"customer": "C3", "demand": 16214}
  ],
  "warehouses": [
    {"warehouse": "S1", "fixed_costs": 102.33},
    {"warehouse": "S2", "fixed_costs": 94.92},
    {"warehouse": "S3", "fixed_costs": 91.83}
  ],
  "transportation_costs": {
    "S1": {"C1": 1506.22, "C2": 70.90, "C3": 8.44},
    "S2": {"C1": 1732.65, "C2": 1780.72, "C3": 567.44},
    "S3": {"C1": 115.66, "C2": 100.76, "C3": 64.68}
  }
}