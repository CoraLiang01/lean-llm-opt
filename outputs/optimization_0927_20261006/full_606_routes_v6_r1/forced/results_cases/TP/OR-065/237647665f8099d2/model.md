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

- $y_i \in \{0,1\}$: $1$ if warehouse $i$ is operational, $0$ otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to customer $j$, for $i \in I$, $j \in J$

##### Mathematical Model

Minimize total cost:
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:

1. **Demand satisfaction** (each customer’s demand must be met):
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]

2. **Facility activation** (no supply from a warehouse unless it is open):
   \[
   x_{ij} \leq d_j y_i \qquad \forall i \in I,\, \forall j \in J
   \]

3. **Variable domains**:
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, \forall j \in J
   \]

##### Data (retrieved)

- $I = \{S1, S2, S3\}$
- $J = \{C1, C2, C3\}$
- $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.90$, $c_{S1,C3} = 8.44$
- $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
- $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$