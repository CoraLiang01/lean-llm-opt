##### Sets

- $I = \{B1, B2, B3, B4, B5, B6, B7, B8\}$: set of candidate depots.
- $J = \{Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10\}$: set of service zones.

##### Parameters

- Opening costs:
  - $c_{B1} = 11$
  - $c_{B2} = 14$
  - $c_{B3} = 10$
  - $c_{B4} = 13$
  - $c_{B5} = 16$
  - $c_{B6} = 9$
  - $c_{B7} = 12$
  - $c_{B8} = 15$

- Coverage sets (depots covering each zone):

  - $S_{Z1} = \{B1, B4\}$
  - $S_{Z2} = \{B1, B2\}$
  - $S_{Z3} = \{B2, B5\}$
  - $S_{Z4} = \{B3, B7\}$
  - $S_{Z5} = \{B1, B3, B8\}$
  - $S_{Z6} = \{B2, B4, B8\}$
  - $S_{Z7} = \{B4, B5\}$
  - $S_{Z8} = \{B3, B6\}$
  - $S_{Z9} = \{B5, B6, B8\}$
  - $S_{Z10} = \{B6, B7\}$

##### Decision Variables

- $y_i \in \{0,1\}$ for $i \in I$: $y_i = 1$ if depot $i$ is opened, $0$ otherwise.

##### Objective Function

\[
\min \sum_{i \in I} c_i y_i
\]

##### Constraints

1. Coverage: For each service zone $j \in J$, at least one depot covering $j$ must be opened:
   - For $Z1$: $y_{B1} + y_{B4} \geq 1$
   - For $Z2$: $y_{B1} + y_{B2} \geq 1$
   - For $Z3$: $y_{B2} + y_{B5} \geq 1$
   - For $Z4$: $y_{B3} + y_{B7} \geq 1$
   - For $Z5$: $y_{B1} + y_{B3} + y_{B8} \geq 1$
   - For $Z6$: $y_{B2} + y_{B4} + y_{B8} \geq 1$
   - For $Z7$: $y_{B4} + y_{B5} \geq 1$
   - For $Z8$: $y_{B3} + y_{B6} \geq 1$
   - For $Z9$: $y_{B5} + y_{B6} + y_{B8} \geq 1$
   - For $Z10$: $y_{B6} + y_{B7} \geq 1$

2. Binary restrictions:
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\min\ & 11y_{B1} + 14y_{B2} + 10y_{B3} + 13y_{B4} + 16y_{B5} + 9y_{B6} + 12y_{B7} + 15y_{B8} \\
\text{s.t.}\quad
& y_{B1} + y_{B4} \geq 1 \\
& y_{B1} + y_{B2} \geq 1 \\
& y_{B2} + y_{B5} \geq 1 \\
& y_{B3} + y_{B7} \geq 1 \\
& y_{B1} + y_{B3} + y_{B8} \geq 1 \\
& y_{B2} + y_{B4} + y_{B8} \geq 1 \\
& y_{B4} + y_{B5} \geq 1 \\
& y_{B3} + y_{B6} \geq 1 \\
& y_{B5} + y_{B6} + y_{B8} \geq 1 \\
& y_{B6} + y_{B7} \geq 1 \\
& y_i \in \{0,1\} \quad \forall i \in \{B1, B2, B3, B4, B5, B6, B7, B8\}
\end{align*}
\]