##### Objective Function:

$\quad \min \left( \sum_{i=1}^3 f_i y_i + \sum_{i=1}^3 \sum_{j=1}^3 c_{ij} x_{ij} \right)$

where:
- $f_i$ is the fixed cost of opening warehouse $S_i$
- $c_{ij}$ is the transportation cost per unit from warehouse $S_i$ to customer $C_j$
- $y_i$ is a binary variable indicating if warehouse $S_i$ is open ($y_i \in \{0,1\}$)
- $x_{ij}$ is the quantity supplied from warehouse $S_i$ to customer $C_j$

##### Constraints

###### 1. Demand Satisfaction:

$\sum_{i=1}^3 x_{ij} = d_j \quad \forall j \in \{1,2,3\}$

where $d_j$ is the demand of customer $C_j$.

###### 2. Warehouse Activation:

$x_{ij} \leq d_j y_i \quad \forall i \in \{1,2,3\},\ \forall j \in \{1,2,3\}$

###### 3. Variable Domains:

$y_i \in \{0,1\} \quad \forall i \in \{1,2,3\}$

$x_{ij} \geq 0 \quad \forall i \in \{1,2,3\},\ \forall j \in \{1,2,3\}$

##### Retrieved Information

{
  "warehouses": [
    "S1",
    "S2",
    "S3"
  ],
  "customers": [
    "C1",
    "C2",
    "C3"
  ],
  "fixed_costs": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "demands": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "transportation_costs": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.90,
      "C3": 8.44
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68
    }
  }
}

##### Full Model with Parameters

Let $i \in \{1,2,3\}$ index warehouses $S1, S2, S3$ and $j \in \{1,2,3\}$ index customers $C1, C2, C3$.

- $f_1 = 102.33$, $f_2 = 94.92$, $f_3 = 91.83$
- $d_1 = 1083$, $d_2 = 776$, $d_3 = 16214$
- $c_{ij}$ matrix:

\[
\begin{array}{c|ccc}
 & C1 & C2 & C3 \\
\hline
S1 & 1506.22 & 70.90 & 8.44 \\
S2 & 1732.65 & 1780.72 & 567.44 \\
S3 & 115.66 & 100.76 & 64.68 \\
\end{array}
\]

The model:

\[
\min \Bigg[
102.33\,y_1 + 94.92\,y_2 + 91.83\,y_3
+ 1506.22\,x_{1,1} + 70.90\,x_{1,2} + 8.44\,x_{1,3}
+ 1732.65\,x_{2,1} + 1780.72\,x_{2,2} + 567.44\,x_{2,3}
+ 115.66\,x_{3,1} + 100.76\,x_{3,2} + 64.68\,x_{3,3}
\Bigg]
\]

subject to

\[
\begin{align*}
x_{1,1} + x_{2,1} + x_{3,1} &= 1083 \\
x_{1,2} + x_{2,2} + x_{3,2} &= 776 \\
x_{1,3} + x_{2,3} + x_{3,3} &= 16214 \\
x_{i,j} &\leq d_j y_i \quad \forall i,j \\
y_i &\in \{0,1\} \quad \forall i \\
x_{i,j} &\geq 0 \quad \forall i,j \\
\end{align*}
\]