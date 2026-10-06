##### Sets and Parameters

- Let $I = \{B1, B2, B3, B4, B5, B6, B7, B8\}$ be the set of candidate depots.
- Let $J = \{Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10\}$ be the set of service zones.
- Let $f_i$ be the opening cost for depot $i \in I$:
  - $f_{B1} = 11$
  - $f_{B2} = 14$
  - $f_{B3} = 10$
  - $f_{B4} = 13$
  - $f_{B5} = 16$
  - $f_{B6} = 9$
  - $f_{B7} = 12$
  - $f_{B8} = 15$
- Let $C_i$ be the set of service zones covered by depot $i$:
  - $C_{B1} = \{Z1, Z2, Z5\}$
  - $C_{B2} = \{Z2, Z3, Z6\}$
  - $C_{B3} = \{Z4, Z5, Z8\}$
  - $C_{B4} = \{Z1, Z6, Z7\}$
  - $C_{B5} = \{Z3, Z7, Z9\}$
  - $C_{B6} = \{Z8, Z9, Z10\}$
  - $C_{B7} = \{Z4, Z10\}$
  - $C_{B8} = \{Z5, Z6, Z9\}$
- For each zone $j \in J$, let $I_j = \{i \in I : j \in C_i\}$ be the set of depots that cover zone $j$:
  - $I_{Z1} = \{B1, B4\}$
  - $I_{Z2} = \{B1, B2\}$
  - $I_{Z3} = \{B2, B5\}$
  - $I_{Z4} = \{B3, B7\}$
  - $I_{Z5} = \{B1, B3, B8\}$
  - $I_{Z6} = \{B2, B4, B8\}$
  - $I_{Z7} = \{B4, B5\}$
  - $I_{Z8} = \{B3, B6\}$
  - $I_{Z9} = \{B5, B6, B8\}$
  - $I_{Z10} = \{B6, B7\}$

##### Decision Variables

$y_i \in \{0,1\}$: $1$ if depot $i$ is opened, $0$ otherwise, for all $i \in I$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i
\]

##### Constraints

For each service zone $j \in J$:
\[
\sum_{i \in I_j} y_i \geq 1 \qquad \forall j \in J
\]

##### Variable Domains

\[
y_i \in \{0,1\} \qquad \forall i \in I
\]

##### Full Model

\[
\begin{align*}
\min\quad & 11y_{B1} + 14y_{B2} + 10y_{B3} + 13y_{B4} + 16y_{B5} + 9y_{B6} + 12y_{B7} + 15y_{B8} \\
\text{s.t.}\quad
& y_{B1} + y_{B4} \geq 1 \quad \text{(cover } Z1) \\
& y_{B1} + y_{B2} \geq 1 \quad \text{(cover } Z2) \\
& y_{B2} + y_{B5} \geq 1 \quad \text{(cover } Z3) \\
& y_{B3} + y_{B7} \geq 1 \quad \text{(cover } Z4) \\
& y_{B1} + y_{B3} + y_{B8} \geq 1 \quad \text{(cover } Z5) \\
& y_{B2} + y_{B4} + y_{B8} \geq 1 \quad \text{(cover } Z6) \\
& y_{B4} + y_{B5} \geq 1 \quad \text{(cover } Z7) \\
& y_{B3} + y_{B6} \geq 1 \quad \text{(cover } Z8) \\
& y_{B5} + y_{B6} + y_{B8} \geq 1 \quad \text{(cover } Z9) \\
& y_{B6} + y_{B7} \geq 1 \quad \text{(cover } Z10) \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]