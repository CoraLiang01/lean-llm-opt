##### Sets

- Suppliers: $I = \{S1, S2, S3, S4, S5\}$
- Branches (customers): $J = \{C1, C2, C3, C4, C5\}$

##### Parameters

- Demand for each branch:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$
- Fixed cost for each supplier:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$
- Transportation costs per unit from each supplier to each branch:

\[
\begin{array}{c|ccccc}
 & C1 & C2 & C3 & C4 & C5 \\
\hline
S1 & 150.74 & 0.02 & 49.13 & 2080.15 & 426.4 \\
S2 & 233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
S3 & 55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
S4 & 1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
S5 & 1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
\end{array}
\]

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational, 0 otherwise, for $i \in I$
- $x_{ij} \geq 0$: quantity supplied from supplier $i$ to branch $j$, for $i \in I$, $j \in J$

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

where $c_{ij}$ is the transportation cost per unit from supplier $i$ to branch $j$.

##### Constraints

1. **Demand satisfaction** (each branch's demand must be met):

   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supplier activation** (a supplier can only supply if it is open):

   \[
   x_{ij} \leq M_{ij} y_i \qquad \forall i \in I,\, j \in J
   \]

   where $M_{ij}$ is a sufficiently large constant, e.g., $M_{ij} \geq d_j$.

3. **Variable domains**:

   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Complete Numerical Formulation

\[
\begin{align*}
\min\ & 97.65\,y_{S1} + 99.76\,y_{S2} + 100.76\,y_{S3} + 105.32\,y_{S4} + 98.88\,y_{S5} \\
&+ 150.74\,x_{S1,C1} + 0.02\,x_{S1,C2} + 49.13\,x_{S1,C3} + 2080.15\,x_{S1,C4} + 426.4\,x_{S1,C5} \\
&+ 233.05\,x_{S2,C1} + 97.73\,x_{S2,C2} + 49.84\,x_{S2,C3} + 1982.39\,x_{S2,C4} + 23.96\,x_{S2,C5} \\
&+ 55.68\,x_{S3,C1} + 935.61\,x_{S3,C2} + 4.03\,x_{S3,C3} + 73.09\,x_{S3,C4} + 525.32\,x_{S3,C5} \\
&+ 1483.82\,x_{S4,C1} + 1801.08\,x_{S4,C2} + 112.16\,x_{S4,C3} + 816.05\,x_{S4,C4} + 107.01\,x_{S4,C5} \\
&+ 1119.47\,x_{S5,C1} + 884.31\,x_{S5,C2} + 0.08\,x_{S5,C3} + 1544.95\,x_{S5,C4} + 543.67\,x_{S5,C5}
\end{align*}
\]

Subject to:

\[
\begin{align*}
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} + x_{S5,C1} &\geq 143 \\
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} + x_{S5,C2} &\geq 6 \\
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} + x_{S5,C3} &\geq 10 \\
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} + x_{S5,C4} &\geq 25 \\
x_{S1,C5} + x_{S2,C5} + x_{S3,C5} + x_{S4,C5} + x_{S5,C5} &\geq 3 \\
\end{align*}
\]

\[
\begin{align*}
x_{ij} \leq d_j\, y_i \qquad \forall i \in I,\, j \in J
\end{align*}
\]

\[
y_i \in \{0,1\} \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

##### All identifiers and coefficients are preserved as retrieved.