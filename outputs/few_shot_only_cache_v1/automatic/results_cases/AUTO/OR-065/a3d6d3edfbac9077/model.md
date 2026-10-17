##### Parameters

- Warehouses (facilities): $F = \{S1, S2, S3\}$
- Musicians/bands (customers): $S = \{C1, C2, C3\}$
- Demand for each musician/band:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
- Fixed cost for each warehouse:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
- Transportation cost per unit from each warehouse to each musician/band:
  - $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.90$, $c_{S1,C3} = 8.44$
  - $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
  - $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- Let $M = \sum_{j \in S} d_j = 1083 + 776 + 16214 = 18073$

##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in F$ to musician/band $j \in S$ (continuous)
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary)

##### Objective Function

\[
\min \sum_{i \in F} \sum_{j \in S} c_{ij} x_{ij} + \sum_{i \in F} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:** Each musician/band must receive exactly its demand.
   \[
   \sum_{i \in F} x_{ij} = d_j, \quad \forall j \in S
   \]
2. **Warehouse activation:** No shipments from inactive warehouses.
   \[
   \sum_{j \in S} x_{ij} \leq M y_i, \quad \forall i \in F
   \]
3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters in full:

- $F = \{S1, S2, S3\}$
- $S = \{C1, C2, C3\}$
- $d_{C1} = 1083$, $d_{C2} = 776$, $d_{C3} = 16214$
- $f_{S1} = 102.33$, $f_{S2} = 94.92$, $f_{S3} = 91.83$
- $c_{S1,C1} = 1506.22$, $c_{S1,C2} = 70.90$, $c_{S1,C3} = 8.44$
- $c_{S2,C1} = 1732.65$, $c_{S2,C2} = 1780.72$, $c_{S2,C3} = 567.44$
- $c_{S3,C1} = 115.66$, $c_{S3,C2} = 100.76$, $c_{S3,C3} = 64.68$
- $M = 18073$

##### Full Mathematical Model

\[
\begin{align*}
\min \quad & 1506.22\,x_{S1,C1} + 70.90\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
           & + 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
           & + 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} \\
           & + 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\[2ex]
\text{s.t.} \quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\[2ex]
& x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\,y_{S2} \\
& x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\,y_{S3} \\[2ex]
& x_{ij} \geq 0 \quad \forall i \in F,\, j \in S \\
& y_i \in \{0,1\} \quad \forall i \in F
\end{align*}
\]